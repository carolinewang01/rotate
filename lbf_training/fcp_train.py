"""
Based on PureJaxRL Implementation of PPO. 
Script adapted from JaxMARL IPPO RNN Smax script.
"""
import os
from datetime import datetime
import jax
import jax.numpy as jnp
import functools
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from jaxmarl.wrappers.baselines import LogWrapper
import jaxmarl
import jumanji
import wandb
import pickle

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from lbf_training.ippo_checkpoints import make_train, ActorCritic, unbatchify

def train_partners_in_parallel(config):
    '''
    Train a pool of partners for FCP. Return checkpoints for all partners.
    '''
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config)))
        out = train_jit(rngs)
    
    return out['checkpoints'], out['metrics']

def save_checkpoints(config, checkpoints):
    curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(config["RESULTS_PATH"], curr_datetime) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    savepath = f"{save_dir}/checkpoint.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(checkpoints, f)
    return savepath

def load_checkpoints(path):
    with open(path, "rb") as f:
        checkpoints = pickle.load(f)
    return checkpoints

def train_fcp_agent(config, checkpoints):
    '''
    Train an FCP agent using the given partner checkpoints and IPPO.
    Return model checkpoints and metrics. 
    '''
    # ------------------------------
    # 1) Flatten partner checkpoints into shape (N, ...) if desired
    #    but we can also keep them as (n_seeds, m_ckpts, ...).
    #    We'll just do gather via dynamic indexing in a jittable way.
    # ------------------------------
    partner_params = checkpoints["params"]  # This is the full PyTree
    n_seeds = partner_params["Dense_0"]["kernel"].shape[0]
    m_ckpts = partner_params["Dense_0"]["kernel"].shape[1]
    num_total_partners = n_seeds * m_ckpts

    # We can define a small helper to gather the correct slice for each environment
    # from shape (n_seeds, m_ckpts, ...) -> (num_envs, ...)
    # We'll do an integer mapping from [0, num_total_partners) -> (seed_idx, ckpt_idx).
    def unravel_partner_idx(idx):
        """Given a scalar in [0, n_seeds*m_ckpts), return (seed_idx, ckpt_idx)."""
        # seed_idx = idx // m_ckpts
        # ckpt_idx = idx % m_ckpts
        # We'll do jax-friendly approach:
        seed_idx = jnp.floor_divide(idx, m_ckpts)
        ckpt_idx = jnp.mod(idx, m_ckpts)
        return seed_idx, ckpt_idx

    def gather_partner_params(partner_params_pytree, idx_vec):
        """
        idx_vec shape: (num_envs,) each in [0, n_seeds*m_ckpts).
        Return a new pytree with shape (num_envs, ...) for each leaf.
        """
        # We'll define a function that gathers from each leaf
        # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
        # We'll vmap a slicing function.
        def gather_leaf(leaf):
            # leaf shape: (n_seeds, m_ckpts, ...)
            # We'll define a function that slices out a single index:
            def slice_one(idx):
                s, c = unravel_partner_idx(idx)
                return leaf[s, c]  # shape (...)
            return jax.vmap(slice_one)(idx_vec)

        return jax.tree.map(gather_leaf, partner_params_pytree)

    # ------------------------------
    # 2) Prepare environment (same as IPPO).
    #    We'll assume exactly 2 agents: agent_0 = trainable, agent_1 = partner.
    # ------------------------------
    if config["ENV_NAME"] == 'lbf':
        env = jumanji.make('LevelBasedForaging-v0')
        env = JumanjiToJaxMARL(env)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    num_agents = env.num_agents
    assert num_agents == 2, "This FCP snippet assumes exactly 2 agents."

    # ------------------------------
    # 3) Build the FCP training function, closely mirroring `make_train(...)`.
    # ------------------------------
    def make_fcp_train(config, env, partner_params):
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]
        # If you rely on the LogWrapper for metrics like "returned_episode_returns", wrap again:
        env = LogWrapper(env)

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        class Transition(NamedTuple):
            done: jnp.ndarray
            action: jnp.ndarray
            value: jnp.ndarray
            reward: jnp.ndarray
            log_prob: jnp.ndarray
            obs: jnp.ndarray
            info: jnp.ndarray

        def train(rng):
            # --------------------------
            # 3a) Init agent_0 network
            # --------------------------
            agent0_net = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
            rng, init_rng = jax.random.split(rng)
            dummy_obs = jnp.zeros(env.observation_space(env.agents[0]).shape)
            init_params = agent0_net.init(init_rng, dummy_obs)

            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=agent0_net.apply,
                params=init_params,
                tx=tx,
            )

            # --------------------------
            # 3b) Init envs & partner indices
            # --------------------------
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            partner_indices = jax.random.randint(
                key=partner_rng,
                shape=(config["NUM_ENVS"],),
                minval=0,
                maxval=num_total_partners
            )

            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step(runner_state, unused):
                """
                runner_state = (train_state, env_state, last_obs, partner_indices, rng)
                Returns updated runner_state, and a Transition for agent_0.
                """
                train_state, env_state, last_obs, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Agent_0 action
                pi_0, val_0 = agent0_net.apply(train_state.params, obs_0)
                act_0 = pi_0.sample(seed=actor_rng)
                logp_0 = pi_0.log_prob(act_0)

                # Agent_1 (partner) action
                #  gather correct partner params for each env -> shape (num_envs, ...)
                gathered_params = gather_partner_params(partner_params, partner_indices)
                # We'll vmap the partner net apply
                def apply_partner(p, o, rng_):
                    # p: single-partner param dictionary
                    # o: single obs vector
                    # rng_: single environment's RNG
                    pi, _ = ActorCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, o)
                    return pi.sample(seed=rng_)

                rng_partner = jax.random.split(partner_rng, config["NUM_ENVS"])
                act_1 = jax.vmap(apply_partner)(gathered_params, obs_1, rng_partner)

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obsv2, env_state2, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:config["NUM_ENVS"]], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0
                )
                new_runner_state = (train_state, env_state2, obsv2, partner_indices, rng)
                return new_runner_state, transition

            # --------------------------
            # 3d) GAE & update step
            # --------------------------
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, returns = batch_info
                    def _loss_fn(params, traj_batch, gae, target_v):
                        pi, value = agent0_net.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        v_loss_1 = jnp.square(value - target_v)
                        v_loss_2 = jnp.square(value_pred_clipped - target_v)
                        value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_1, v_loss_2))

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm
                        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                        pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(train_state.params, traj_batch, advantages, returns)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux_vals)

                (train_state, traj_batch, advantages, targets, rng) = update_state
                rng, perm_rng = jax.random.split(rng)
                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"] // 2 
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(perm_rng, batch_size)

                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, env_state, last_obs, partner_indices, rng, update_steps) = update_runner_state

                # 1) rollout
                runner_state = (train_state, env_state, last_obs, partner_indices, rng)
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
                (train_state, env_state, last_obs, partner_indices, rng) = runner_state

                # 2) advantage
                obs_batch_final = last_obs["agent_0"]
                # jnp.stack([last_obs[i]["agent_0"].flatten() for i in range(config["NUM_ENVS"])])
                _, last_val = agent0_net.apply(train_state.params, obs_batch_final)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # (Optional) re-sample partner for each env for next rollout
                rng, p_rng = jax.random.split(rng)
                new_partner_idx = jax.random.randint(
                    key=p_rng, shape=(config["NUM_ENVS"],),
                    minval=0, maxval=num_total_partners
                )

                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps

                new_runner_state = (train_state, env_state, last_obs, new_partner_idx, rng, update_steps + 1)
                return (new_runner_state, metric)

            # --------------------------
            # 3e) Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), params_pytree)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, partner_idx, rng, update_steps),
                 checkpoint_array, ckpt_idx) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, partner_idx, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, partner_idx, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

                def store_ckpt(args):
                    ckpt_arr, cidx = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )
                    return (new_ckpt_arr, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx) = jax.lax.cond(
                    to_store, store_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx)
                )

                return ((train_state, env_state, last_obs, partner_idx, rng, update_steps),
                        checkpoint_array, ckpt_idx), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # initial runner state for scanning
            update_steps = 0
            update_runner_state = (train_state, env_state, obsv, partner_indices, rng, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx)

            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx) = state_with_ckpt

            out = {
                "runner_state": final_runner_state,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": {"params": checkpoint_array},
            }
            return out

        return train
    # ------------------------------
    # 4) Actually run the FCP training
    # ------------------------------
    # fcp_train_fn = make_fcp_train(config, env, partner_params)
    rng = jax.random.PRNGKey(config["SEED"])
    # TODO: vmap across multiple seeds. 
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    with jax.disable_jit(False):
        fcp_train_fn = jax.jit(jax.vmap(make_fcp_train(config, env, partner_params)))
        out = fcp_train_fn(rngs)

    return out["checkpoints"], out["metrics"]

if __name__ == "__main__":
    # set hyperparameters:
    config = {
        "LR": 1.e-4,
        "NUM_ENVS": 8, # 16,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": 1e5, # 1e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16, # 4,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.01,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {
        },
        "ANNEAL_LR": True,
        "SEED": 0,
        "NUM_SEEDS": 2,
        "RESULTS_PATH": "results/lbf"
    }
    
    # out_ckpts, out_metrics = train_partners_in_parallel(config)
    # savepath = save_checkpoints(config, out_ckpts)
    # print(f"Saved partner checkpoints to {savepath}")
    #####################################
    ckpt_path = "results/lbf/2025-01-23_20-35-18/checkpoint.pkl"
    out_ckpts = load_checkpoints(ckpt_path)
    #####################################
    out_ckpts, out_metrics = train_fcp_agent(config, out_ckpts)

    # visualize results
    # print results for each seed
    # metrics values shape is (num_seeds, num_updates, num_envs, ???)
    for i in range(config["NUM_SEEDS"]):
        print("Seed: ", i)
        print("Mean Return (Last): ", out_metrics["returned_episode_returns"][i, -1].mean())
        print("Std Return (Last): ", out_metrics["returned_episode_returns"][i, -1].std())

        print("Mean Percent Eaten (Last): ", out_metrics["percent_eaten"][i, -1].mean())
        print("Std Percent Eaten (Last): ", out_metrics["percent_eaten"][i, -1].std())

    for i in range(config["NUM_SEEDS"]):
        xs = out_metrics["update_steps"][i] * config["NUM_ENVS"] * config["NUM_STEPS"]
        ys = out_metrics["percent_eaten"][i].mean((1, 2))
        plt.plot(xs, ys)
    
    plt.xlabel("Time Step")
    plt.ylabel("Percent Eaten")
    plt.show()
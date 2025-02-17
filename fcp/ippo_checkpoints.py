from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import jumanji
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from fcp.networks import ActorCritic
from fcp.vis_utils import get_stats, plot_metrics


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def get_rollout(train_state, config):
    if config["ENV_NAME"] == 'lbf':
        env = jumanji.make('LevelBasedForaging-v0')
        env = JumanjiToJaxMARL(env)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state.params

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a, key_s = jax.random.split(key, 3)
        obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        
        pi, value = network.apply(network_params, obs_batch)
        actions = pi.sample(seed=key_a)

        env_act = unbatchify(actions, env.agents, config["NUM_ENVS"], env.num_agents)
        env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_info(x: dict, agent_list, num_actors):
    '''Handle special case that info has both per-agent and global information'''
    x = jnp.stack([x[a] for a in x if a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["ENV_NAME"] == 'lbf':
        env = jumanji.make('LevelBasedForaging-v0')
        env = JumanjiToJaxMARL(env)
    else: 
        jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

        init_x = init_x.flatten()

        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
                optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info

                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
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

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            
            rng = update_state[-1]
            update_steps += 1
            
                            # TODO: consider whether we need to return NEW env state, last obs, etc. 

            runner_state = (train_state, env_state, last_obs, rng)
            return (runner_state, update_steps), metric

        checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
        num_ckpts = config["NUM_CHECKPOINTS"]

        # (3) Build a PyTree that can hold the parameters for all checkpoints.
        #     For each leaf x in train_state.params, create an array of shape
        #     (num_ckpts,) + x.shape to hold all saved parameter states.
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            # update_runner_state is ((train_state, env_state, obsv, rng), update_steps)
            # Run one PPO update step
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state

            # Decide if we store a checkpoint now
            # condition: step % checkpoint_interval == 0
            # (You can tweak so you store at the end, or start, etc.)
            to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

            def store_ckpt_fn(args):
                # Write current runner_state[0].params into checkpoint_array at ckpt_idx
                # and increment ckpt_idx
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args  # No changes if we don't store

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, # if to_store, execute true function(operand). else, execute false function(operand).
                store_ckpt_fn, # true fn
                skip_ckpt_fn, # false fn
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # (5) Use lax.scan over NUM_UPDATES
        rng, _rng = jax.random.split(rng)
        update_steps = 0
        update_runner_state = ((train_state, env_state, obsv, _rng), update_steps)
        checkpoint_array = init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,  # No per-step input data
            length=config["NUM_UPDATES"],
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array
        }
    return train

if __name__ == "__main__":

    # set hyperparameters:
    config = {
        "LR": 1.e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 15,
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
        # "layout" : "cramped_room"
        },
        "ANNEAL_LR": True,
        "SEED": 0,
        "NUM_SEEDS": 3
    }

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config)))
        out = train_jit(rngs)


    # out['checkpoints']['params']['Dense_0']['kernel'] has shape (num_seeds, num_ckpts, *param_shape)
    # metrics values shape is (num_seeds, num_updates, num_rollout_steps, num_envs*num_agents)
    all_stats = get_stats(out['metrics'], stats=("percent_eaten", "returned_episode_returns"), num_envs=config["NUM_ENVS"])
    plot_metrics(all_stats, config["NUM_SEEDS"], 
                 config["NUM_UPDATES"], config["NUM_STEPS"], config["NUM_ENVS"])
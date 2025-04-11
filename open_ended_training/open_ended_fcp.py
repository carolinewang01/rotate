'''
An implementation of open-ended FCP where both the ego agent and all 
partner agents are MLP actor critics.
'''
import logging
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
import time
from envs import make_env
from open_ended_training.train_ippo import make_train, unbatchify, Transition
from common.mlp_actor_critic import ActorCritic
from common.wandb_visualizations import Logger

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_partners_in_parallel(config, partner_rng, env):
    '''
    Train a pool of partners for FCP using IPPO w/parameter sharing. 
    Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
    '''
    rngs = jax.random.split(partner_rng, config["PARTNER_POP_SIZE"])
    train_jit = jax.jit(jax.vmap(make_train(config, env)))
    out = train_jit(rngs)
    return out

def train_fcp_agent(config, checkpoints, init_fcp_params, fcp_rng, env):
    '''
    Train an FCP agent using the given partner checkpoints and IPPO.
    Return model checkpoints, final parameters, and metrics. 
    '''
    partner_params = checkpoints["params"]  # This is the full PyTree
    n_seeds, m_ckpts = partner_params["Dense_0"]["kernel"].shape[:2]
    num_total_partners = n_seeds * m_ckpts

    def unravel_partner_idx(idx):
        """Given a scalar in [0, n_seeds*m_ckpts), return (seed_idx, ckpt_idx)."""
        # seed_idx = idx // m_ckpts
        # ckpt_idx = idx % m_ckpts
        seed_idx = jnp.floor_divide(idx, m_ckpts)
        ckpt_idx = jnp.mod(idx, m_ckpts)
        return seed_idx, ckpt_idx

    def gather_partner_params(partner_params_pytree, idx_vec):
        """
        partner_params_pytree: pytree with all partner params. Each leaf has shape (n_seeds, m_ckpts, ...).
        idx_vec: a vector of indices with shape (num_envs,) each in [0, n_seeds*m_ckpts).

        Return a new pytree where each leaf has shape (num_envs, ...). Each leaf has a sampled
        partner's parameters for each environment.
        """
        # We'll define a function that gathers from each leaf
        # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
        # We'll vmap a slicing function.
        def gather_leaf(leaf):
            def slice_one(idx):
                seed_idx, ckpt_idx = unravel_partner_idx(idx)
                return leaf[seed_idx, ckpt_idx]  # shape (...)
            return jax.vmap(slice_one)(idx_vec)

        return jax.tree.map(gather_leaf, partner_params_pytree)

    def make_fcp_train(config, partner_params):
        # Assume exactly 2 agents: agent_0 = trainable ego, agent_1 = nontrainable partner
        num_agents = env.num_agents
        assert num_agents == 2, "This FCP snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # --------------------------
            # 3a) Init agent_0 network
            # --------------------------
            agent0_net = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
            init_params = init_fcp_params

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

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 action
                agent0_input = (obs_0, avail_actions_0)
                pi_0, val_0 = agent0_net.apply(train_state.params, agent0_input)
                act_0 = pi_0.sample(seed=actor_rng)
                logp_0 = pi_0.log_prob(act_0)

                # Agent_1 (partner) action
                # Gather correct partner params for each env -> shape (num_envs, ...)
                # Note that partner idxs are resampled after every update
                gathered_params = gather_partner_params(partner_params, partner_indices)
                # We'll vmap the partner net apply
                def apply_partner(p, input_x, rng_):
                    # p: single-partner param dictionary
                    # input_x: single input to partner network
                    # rng_: single environment's RNG
                    pi, _ = ActorCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, input_x)
                    return pi.sample(seed=rng_)

                rng_partner = jax.random.split(partner_rng, config["NUM_ENVS"])
                act_1 = jax.vmap(apply_partner)(gathered_params, (obs_1, avail_actions_1), rng_partner)

                # Combine actions into the env format
                combined_actions = jnp.concatenate([act_0, act_1], axis=0)  # shape (2*num_envs,)
                env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step env
                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                obs_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )
                # note that num_actors = num_envs * num_agents
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, partner_indices, rng)
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
                        pi, value = agent0_net.apply(params, (traj_batch.obs, traj_batch.avail_actions))
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        pg_loss_1 = ratio * gae_norm
                        pg_loss_2 = jnp.clip(
                            ratio, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm
                        pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))

                        # Entropy
                        entropy = jnp.mean(pi.entropy())

                        total_loss = pg_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, pg_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state.params, traj_batch, advantages, returns)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux_vals)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"] // 2 
                assert (
                    batch_size == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
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
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, last_obs, partner_indices, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                input_0 = (
                    last_obs_batch_0, jax.lax.stop_gradient(avail_actions_0)
                )
                _, last_val = agent0_net.apply(train_state.params, input_0)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, _ = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # Re-sample partner for each env for next rollout
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
            # 3e) PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

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
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
            }
            return out

        return train
    # ------------------------------
    # 4) Run the FCP training
    # ------------------------------
    out = make_fcp_train(config, partner_params)(fcp_rng)
    return out

def open_ended_training_step(carry, config, env):
    prev_fcp_params, rng = carry

    rng, partner_rng, train_fcp_rng = jax.random.split(rng, 3)
    # train partner agents
    train_partner_out = train_partners_in_parallel(config, partner_rng, env)
    train_partner_ckpts = train_partner_out["checkpoints"]

    # train FCP agent
    fcp_out = train_fcp_agent(config, train_partner_ckpts, prev_fcp_params, train_fcp_rng, env)
    updated_fcp_parameters = fcp_out["final_params"]

    carry = (updated_fcp_parameters, rng)
    return carry, (train_partner_out, fcp_out)

def initialize_agent(config, rng):
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    agent0_net = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    rng, init_rng = jax.random.split(rng)
    init_x = (
        # init obs, avail_actions
        jnp.zeros(env.observation_space(env.agents[0]).shape),
        jnp.ones(env.action_space(env.agents[0]).shape)
    )
    init_params = agent0_net.init(init_rng, init_x)

    return init_params

def run_fcp(config):
    '''
    Run the open-ended FCP training loop.
    '''
    algorithm_config = dict(config["algorithm"])
    # logger = Logger(config)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, algorithm_config, env)
    
    # Define initial carry
    init_params = initialize_agent(algorithm_config, init_rng)
    init_carry = (init_params, train_rng)

    log.info("Starting open-ended FCP training...")
    start_time = time.time()
    with jax.disable_jit(False):
        final_params, outs = jax.lax.scan(
            open_ended_step_fn, 
            init_carry, 
            xs=None,
            length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
        )
    
    end_time = time.time()
    log.info(f"Open-ended FCP training completed in {end_time - start_time} seconds.")

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

from envs import make_env
from ppo.ippo import unbatchify, Transition
from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.wandb_visualizations import Logger


def train_adversarial_partners(config, ego_policy, minimax_env):
    '''
    Train an FCP agent using the given partner checkpoints and IPPO.
    Return model checkpoints and metrics. 
    '''

    def make_fcp_train(config):
        # ------------------------------
        # 2) Prepare environment (same as IPPO).
        #    We'll assume exactly 2 agents: agent_0 = trainable, agent_1 = partner.

        # ------------------------------
        env = minimax_env

        num_agents = env.num_agents
        assert num_agents == 2, "This FCP snippet assumes exactly 2 agents."

        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # Assume only controlling 1 confederate
        config["NUM_ACTORS"] = env.num_agents * config["NUM_CONTROLLED_ACTORS"]
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_CONTROLLED_ACTORS"]
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # --------------------------
            # 3a) Init agent_0 network
            # --------------------------
            agent0_net = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
            rng, init_rng = jax.random.split(rng)
            init_x = ( # init obs, avail_actions
                jnp.zeros(env.observation_space(env.agents[0]).shape),
                jnp.ones(env.action_space(env.agents[0]).n),
            )
            
            init_params = agent0_net.init(init_rng, init_x)

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

            # S5 specific parameters
            d_model = config["S5_D_MODEL"]
            ssm_size = config["S5_SSM_SIZE"]
            n_layers = config["S5_N_LAYERS"]
            blocks = config["S5_BLOCKS"]
            block_size = int(ssm_size / blocks)

            Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
            block_size = block_size // 2
            ssm_size = ssm_size // 2
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Vinv = V.conj().T

            ssm_init_fn = init_S5SSM(H=d_model,
                                    P=ssm_size,
                                    Lambda_re_init=Lambda.real,
                                    Lambda_im_init=Lambda.imag,
                                    V=V,
                                    Vinv=Vinv)
            
            partner_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       config=config, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],)
            # --------------------------
            # 3b) Init envs
            # --------------------------
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step(runner_state, unused):
                """
                runner_state = (train_state, env_state, last_obs, rng)
                Returns updated runner_state, and a Transition for agent_0.
                """
                train_state, env_state, last_obs, last_dones, last_partner_h, rng = runner_state
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

                rnn_input_1 = (
                    obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    last_dones.reshape(1, config["NUM_CONTROLLED_ACTORS"]), 
                    avail_actions_1
                )

                # Agent_0 action
                hstate_ego, pi_1, val_1 = partner_net.apply(
                    {'params': ego_policy["params"]}, last_partner_h, 
                    rnn_input_1
                )
                act_1 = pi_1.sample(seed=actor_rng).squeeze()
                logp_1 = pi_1.log_prob(act_0).squeeze()
                val_1 = val_1.squeeze()

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
                    reward=-reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state, env_state_next, obs_next, done["agent_1"], hstate_ego, rng)
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
                    reward = reward.reshape(-1)
                    value = value.reshape(-1)
                    done = done.reshape(-1)

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
                (train_state, env_state, last_obs, last_dones, hstate_ego, rng, update_steps) = update_runner_state

                # 1) rollout
                runner_state = (train_state, env_state, last_obs, last_dones, hstate_ego, rng)
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["NUM_STEPS"])
                (train_state, env_state, last_obs, last_dones, hstate_ego, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # jnp.stack([last_obs[i]["agent_0"].flatten() for i in range(config["NUM_ENVS"])])
                dummy_act_mask = jnp.ones((last_obs_batch_0.shape[0], env.action_space(env.agents[0]).n))
                _, last_val = agent0_net.apply(train_state.params, (last_obs_batch_0, dummy_act_mask))
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["value_losses"] = all_losses[1][0]
                metric["pg_losses"] = all_losses[1][1]
                metric["entropy_losses"] = all_losses[1][2]
                new_runner_state = (train_state, env_state, last_obs, last_dones, hstate_ego, rng, update_steps + 1)
                return (new_runner_state, metric)

            # --------------------------
            # 3e) PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)
            
            def run_single_episode(rng, fcp_param, partner_param):
                # Reset the env.
                rng, reset_rng = jax.random.split(rng)
                obs, env_state = env.reset(reset_rng)

                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Do one step to get a dummy info structure.
                rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                pi0, _ = agent0_net.apply(partner_param, (obs["agent_0"], avail_actions_0))
                # for LBF, IPPO policies do better when sampled than when taking the mode. 
                act0 = pi0.sample(seed=act_rng)

                hstate_ego = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
                prev_done = jnp.zeros(1, dtype=bool)
                rnn_input_1 = (
                    obs["agent_1"].reshape(1, 1, -1),
                    prev_done.reshape(1, 1), 
                    avail_actions_1.reshape(1, -1)
                )

                ego_agent_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       config=config, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],)
                
                hstate_ego, pi1, _ = ego_agent_net.apply({'params': fcp_param}, hstate_ego, rnn_input_1)
                act1 = pi1.sample(seed=part_rng).squeeze()
                    
                both_actions = [act0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                _, _, _, dones, dummy_info = env.step(step_rng, env_state, env_act)
                done_flag = dones["__all__"]

                # We'll use a scan to iterate steps until the episode is done.
                ep_ts = 1
                init_carry = (ep_ts, env_state, obs, rng, done_flag, hstate_ego, dummy_info)
                def scan_step(carry, _):
                    def take_step(carry_step):
                        ep_ts, env_state, obs, rng, done_flag, hstate_ego, last_info = carry_step
                        # Get available actions for agent 0 from environment state
                        avail_actions = env.get_avail_actions(env_state.env_state)
                        avail_actions = jax.lax.stop_gradient(avail_actions)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                        
                        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                        pi0, _ = agent0_net.apply(partner_param, (obs["agent_0"], avail_actions_0))
                        act0 = pi0.sample(seed=act_rng) # sample because mode does worse on LBF

                        rnn_input_1 = (
                            obs["agent_1"].reshape(1, 1, -1),
                            done_flag.reshape(1, 1), 
                            avail_actions_1.reshape(1, -1)
                        )
                        hstate_ego, pi1, _ = ego_agent_net.apply({'params': fcp_param}, hstate_ego, rnn_input_1)
                        act1 = pi1.sample(seed=part_rng).squeeze()
                        both_actions = [act0, act1]
                        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                        obs_next, env_state_next, reward, done, info = env.step(step_rng, env_state, env_act)

                        return (ep_ts + 1, env_state_next, obs_next, rng, done["__all__"], hstate_ego, info)
                            
                    ep_ts, env_state, obs, rng, done_flag, hstate_ego, last_info = carry
                    new_carry = jax.lax.cond(
                        done_flag,
                        # if done, execute true function(operand). else, execute false function(operand).
                        lambda curr_carry: curr_carry, # True fn
                        take_step, # False fn
                        operand=carry
                    )
                    return new_carry, None

                final_carry, _ = jax.lax.scan(
                    scan_step, init_carry, None, length=max_episode_steps)
                # Return the final info (which includes the episode return via LogWrapper).
                return final_carry[-1]

            def run_episodes(rng, fcp_param, partner_param, num_eps):
                def body_fn(carry, _):
                    rng = carry
                    rng, ep_rng = jax.random.split(rng)
                    ep_info = run_single_episode(ep_rng, fcp_param, partner_param)
                    return rng, ep_info
                rng, ep_infos = jax.lax.scan(body_fn, rng, None, length=num_eps)
                return ep_infos  # each leaf has shape (num_eps, ...)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, last_dones, last_hstate_ego, rng, update_steps),
                 checkpoint_array, ckpt_idx, ep_infos) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, last_dones, last_hstate_ego, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, last_dones, last_hstate_ego, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

                def store_and_eval_ckpt(args):
                    gen_eval_rng = jax.random.PRNGKey(config["EVAL_SEED"])
                    ep_infos = run_episodes(gen_eval_rng, ego_policy["params"], train_state.params, max_episode_steps)
                    ckpt_arr_and_ep_infos, cidx = args
                    ckpt_arr, prev_ep_infos = ckpt_arr_and_ep_infos
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )
                    return ((new_ckpt_arr, ep_infos), cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, ckpt_idx) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, ((checkpoint_array, ep_infos), ckpt_idx)
                )
                checkpoint_array, ckpt_infos = checkpoint_array_and_infos

                metric["per_iter_ep_infos"] = ep_infos["returned_episode_returns"]
                return ((train_state, env_state, last_obs, last_dones, last_hstate_ego, rng, update_steps),
                        checkpoint_array, ckpt_idx, ckpt_infos), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # initial runner state for scanning
            update_steps = 0
            gen_eval_rng = jax.random.PRNGKey(config["EVAL_SEED"])
            max_episode_steps = config["MAX_EVAL_EPISODES"]
            ep_infos = run_episodes(gen_eval_rng, ego_policy["params"], train_state.params, max_episode_steps)
            init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)
            init_dones = jnp.ones(config["NUM_CONTROLLED_ACTORS"], dtype=bool)

            update_runner_state = (train_state, env_state, obsv, init_dones, init_hstate_0, rng, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, ep_infos)

            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx, all_ep_infos) = state_with_ckpt

            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "all_ep_infos": metrics["per_iter_ep_infos"]
            }
            return out

        return train
    # ------------------------------
    # 4) Actually run the adversarial teammate training
    # ------------------------------
    # training is vmapped across multiple seeds

    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    with jax.disable_jit(False):
        fcp_train_fn = jax.jit(jax.vmap(make_fcp_train(config)))
        out = fcp_train_fn(rngs)
    return out

def train_fcp_agent(config, checkpoints, fcp_env, init_fcp_params=None):
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
    n_seeds, m_ckpts = partner_params["Dense_0"]["kernel"].shape[:2]
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
        partner_params_pytree: pytree with all partner params. Each leaf has shape (n_seeds, m_ckpts, ...).
        idx_vec: a vector of indices with shape (num_envs,) each in [0, n_seeds*m_ckpts).

        Return a new pytree where each leaf has shape (num_envs, ...). Each leaf has a sampled
        partner's parameters for each environment.
        """
        # We'll define a function that gathers from each leaf
        # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
        # We'll vmap a slicing function.
        def gather_leaf(leaf):
            # leaf shape: (n_seeds, m_ckpts, ...)
            # We'll define a function that slices out a single index:
            def slice_one(idx):
                seed_idx, ckpt_idx = unravel_partner_idx(idx)
                return leaf[seed_idx, ckpt_idx]  # shape (...)
            return jax.vmap(slice_one)(idx_vec)

        return jax.tree.map(gather_leaf, partner_params_pytree)

    # ------------------------------
    # 3) Build the FCP training function, closely mirroring `make_train(...)`.
    # ------------------------------
    def make_fcp_train(config, partner_params):
        # ------------------------------
        # 2) Prepare environment (same as IPPO).
        #    We'll assume exactly 2 agents: agent_0 = trainable, agent_1 = partner.

        # ------------------------------
        env = fcp_env
        env = LogWrapper(env)

        num_agents = env.num_agents
        assert num_agents == 2, "This FCP snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        
        # S5 specific parameters
        d_model = config["S5_D_MODEL"]
        ssm_size = config["S5_SSM_SIZE"]
        n_layers = config["S5_N_LAYERS"]
        blocks = config["S5_BLOCKS"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(H=d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv)

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # --------------------------
            # 3a) Init agent_0 network
            # --------------------------
            agent0_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       config=config, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],)
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

            # Initialize hidden state for RNN
            # init_hstate_0 = ScannedRNN.initialize_carry(config["NUM_CONTROLLED_ACTORS"], config["GRU_HIDDEN_DIM"])
            init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            partner_indices = jax.random.randint(
                key=partner_rng,
                shape=(config["NUM_UNCONTROLLED_ACTORS"],),
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
                train_state, env_state, prev_obs, prev_done, hstate_0, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = prev_obs["agent_0"]
                obs_1 = prev_obs["agent_1"]

                 # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                # obs, done should have shape (sequence_length, num_actors, features) for the RNN
                # hstate should have shape (1, num_actors, hidden_dim)
                rnn_input_0 = (
                    obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    prev_done.reshape(1, config["NUM_CONTROLLED_ACTORS"]), 
                    avail_actions_0
                )

                # Agent_0 action
                # Agent_0 action
                hstate_0, pi_0, val_0 = agent0_net.apply(train_state.params, hstate_0, rnn_input_0)
                act_0 = pi_0.sample(seed=actor_rng).squeeze()
                logp_0 = pi_0.log_prob(act_0).squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action
                # Gather correct partner params for each env -> shape (num_envs, ...)
                # Note that partner idxs are resampled after every update
                gathered_params = gather_partner_params(partner_params, partner_indices)
                # We'll vmap the partner net apply
                def apply_partner(p, o, rng_):
                    # p: single-partner param dictionary
                    # o: single obs vector
                    # rng_: single environment's RNG
                    pi, _ = ActorCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, o)
                    return pi.sample(seed=rng_)

                rng_partner = jax.random.split(partner_rng, config["NUM_UNCONTROLLED_ACTORS"])
                # TODO: verify that the vmap has been performed correctly
                partner_input = (obs_1, avail_actions_1)
                act_1 = jax.vmap(apply_partner)(gathered_params, partner_input, rng_partner)

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
                new_runner_state = (train_state, env_state_next, obs_next, done["agent_0"],  hstate_0, partner_indices, rng)
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
                    init_hstate_0, traj_batch, advantages, returns = batch_info
                    def _loss_fn(params, init_hstate_0, traj_batch, gae, target_v):
                        rnn_input_0 = (
                            traj_batch.obs, # shape (rollout_len, num_actors/num_minibatches, feat_size) =  (128, 4, 15)
                            traj_batch.done, # shape (rollout_len, num_actors/num_minibatches) = (128, 4)
                            traj_batch.avail_actions # shape (rollout_len, num_agents, num_actions) = (128, 4, 6)
                        )
                        _, pi, value = agent0_net.apply(
                            params, 
                            init_hstate_0,
                            rnn_input_0
                        )
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
                        train_state.params, init_hstate_0, traj_batch, advantages, returns)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss_val, aux_vals)

                train_state, init_hstate_0, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                # batch_size is now config["NUM_ENVS"]
                permutation = jax.random.permutation(perm_rng, config["NUM_CONTROLLED_ACTORS"])

                batch = (
                    init_hstate_0, # shape (1, num_agents, hidden_dim) = (1, 16, 64)
                    traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape) = (128, 16, 15)
                    advantages, # shape (rollout_len, num_agents) = (128, 16)
                    targets # shape (rollout_len, num_agents) = (128, 16)
                )

                # each leaf of shuffled batch has shape (rollout_len, num_agents, feat_shape)
                # except for init_hstate_0 which has shape (1, num_agents, hidden_dim)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                # each leaf has shape (num_minibatches, rollout_len, num_agents/num_minibatches, feat_shape)
                # except for init_hstate_0 which has shape (num_minibatches, 1, num_agents/num_minibatches, hidden_dim)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] 
                            + list(x.shape[2:]),
                    ), 1, 0,),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate_0, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_indices, rng, update_steps) = update_runner_state

                # 1) rollout
                runner_state = (train_state, env_state, last_obs, last_done, last_hstate_0, partner_indices, rng)
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["NUM_STEPS"])
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_indices, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                input_0 = (
                    last_obs_batch_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    last_done.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    jax.lax.stop_gradient(avail_actions_0)
                )
                _, _, last_val = agent0_net.apply(train_state.params, last_hstate_0, input_0)
                last_val = last_val.squeeze()
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # 3) PPO update
                update_state = (
                    train_state,
                    init_hstate_0, # shape is (num_controlled_actors, gru_hidden_dim) with all-0s value
                    traj_batch, # obs has shape (rollout_len, num_controlled_actors, -1)
                    advantages,
                    targets,
                    rng
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # Resample partner for each env for next rollout
                # Note that we reset the hidden state after resampling partners by returning init_hstate_0
                rng, p_rng = jax.random.split(rng)
                new_partner_idx = jax.random.randint(
                    key=p_rng, shape=(config["NUM_UNCONTROLLED_ACTORS"],),
                    minval=0, maxval=num_total_partners
                )                
                # Reset environment due to partner change
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = jnp.zeros((config["NUM_CONTROLLED_ACTORS"]), dtype=bool)


                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["value_loss"] = all_losses[1][0]
                metric["actor_loss"] = all_losses[1][1]
                metric["entropy_loss"] = all_losses[1][2]
                new_runner_state = (train_state, env_state, obs, init_done, init_hstate_0, new_partner_idx, rng, update_steps + 1)
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

            max_episode_steps = config["MAX_EVAL_EPISODES"]

            # TODO Gather single episode data
            def run_single_episode(rng, partner_param):
                # Reset the env.
                rng, reset_rng = jax.random.split(rng)
                obs, env_state = env.reset(reset_rng)
                prev_done = jnp.zeros(1, dtype=bool)

                # Do one step to get a dummy info structure.
                rng, act_rng, part_rng, part_idx_rng, step_rng = jax.random.split(rng, 5)
                init_hstate_0 = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)

                # Get agent obses
                obs_0 = obs["agent_0"]
                obs_1 = obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = env.get_avail_actions(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # obs, done should have shape (sequence_length, num_actors, features) for the RNN
                # hstate should have shape (1, num_actors, hidden_dim)
                rnn_input_0 = (
                    obs_0.reshape(1, 1, -1),
                    prev_done.reshape(1, 1), 
                    avail_actions_0
                )

                hstate_0, pi_0, val_0 = agent0_net.apply(train_state.params, init_hstate_0, rnn_input_0)
                act_0 = pi_0.sample(seed=act_rng).squeeze()
                logp_0 = pi_0.log_prob(act_0).squeeze()
                val_0 = val_0.squeeze()

                def apply_partner(p, o, rng_):
                    # p: single-partner param dictionary
                    # o: single obs vector
                    # rng_: single environment's RNG
                    partner_pol, _ = ActorCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, o)
                    return partner_pol.sample(seed=rng_)
                
                partner_input = (obs_1, avail_actions_1)
                act1 = apply_partner(partner_param, partner_input, part_rng)    
                both_actions = [act_0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                _, _, _, dones, dummy_info = env.step(step_rng, env_state, env_act)
                done_flag = dones["__all__"]

                # We'll use a scan to iterate steps until the episode is done.
                ep_ts = 1
                init_carry = (ep_ts, env_state, obs, rng, done_flag, hstate_0, dummy_info)
                def scan_step(carry, _):
                    def take_step(carry_step):
                        ep_ts, env_state, obs, rng, done_flag, hstate_0, last_info = carry_step

                        # Get available actions for agent 0 from environment state
                        avail_actions = env.get_avail_actions(env_state.env_state)
                        avail_actions = jax.lax.stop_gradient(avail_actions)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        rnn_input_0 = (
                            obs_0.reshape(1, 1, -1),
                            prev_done.reshape(1, 1), 
                            avail_actions_0
                        )

                        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                        hstate_0, pi_0, val_0 = agent0_net.apply(train_state.params, hstate_0, rnn_input_0)
                        act_0 = pi_0.sample(seed=act_rng).squeeze()
                        logp_0 = pi_0.log_prob(act_0).squeeze()
                        val_0 = val_0.squeeze()

                        # Compute partner's actions
                        partner_input = (obs_1, avail_actions_1)
                        def apply_partner(p, o, rng_):
                            # p: single-partner param dictionary
                            # o: single obs vector
                            # rng_: single environment's RNG
                            partner_pol, _ = ActorCritic(env.action_space(env.agents[1]).n,
                                                activation=config["ACTIVATION"]).apply({'params': p}, o)
                            return partner_pol.sample(seed=rng_)
                        
                        act1 = apply_partner(partner_param, partner_input, part_rng) 
                        both_actions = [act_0, act1]
                        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                        obs_next, env_state_next, reward, done, info = env.step(step_rng, env_state, env_act)

                        return (ep_ts + 1, env_state_next, obs_next, rng, done["__all__"], hstate_0, info)
                            
                    ep_ts, env_state, obs, rng, done_flag, hstate_0, last_info = carry
                    new_carry = jax.lax.cond(
                        done_flag,
                        # if done, execute true function(operand). else, execute false function(operand).
                        lambda curr_carry: curr_carry, # True fn
                        take_step, # False fn
                        operand=carry
                    )
                    return new_carry, None

                final_carry, _ = jax.lax.scan(
                    scan_step, init_carry, None, length=max_episode_steps)
                # Return the final info (which includes the episode return via LogWrapper).
                return final_carry[-1]
            
            def run_episodes(rng, partner_param, num_eps):
                def body_fn(carry, _):
                    rng = carry
                    rng, ep_rng = jax.random.split(rng)
                    ep_info = run_single_episode(ep_rng, partner_param)
                    return rng, ep_info
                rng, ep_infos = jax.lax.scan(body_fn, rng, None, length=num_eps)
                return ep_infos  # each leaf has shape (num_eps, ...)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                 checkpoint_array, ckpt_idx, init_eval_info) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs,  last_done, hstate_0, partner_idx, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs,  last_done, hstate_0, partner_idx, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

                def store_ckpt(args):
                    ckpt_arr, cidx, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = gather_partner_params(partner_params, eval_partner_indices)
                    eval_rng = jax.random.PRNGKey(config["EVAL_SEED"])
                    eval_ep_return_infos = jax.vmap(lambda x: run_episodes(eval_rng, x, max_episode_steps))(gathered_params)
                    # run_episodes(eval_r
                    # ng, gathered_params[0], max_episode_steps)
                    return (new_ckpt_arr, cidx + 1, eval_ep_return_infos)

                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, ep_ret_infos) = jax.lax.cond(
                    to_store, store_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, init_eval_info)
                )

                metric["per_iter_ep_infos"] = ep_ret_infos["returned_episode_returns"]
                return ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                        checkpoint_array, ckpt_idx, ep_ret_infos), metric

            # TODO Until here
            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # Init eval
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = gather_partner_params(partner_params, eval_partner_indices)
            eval_rng = jax.random.PRNGKey(config["EVAL_SEED"])
            eval_ep_return_infos = jax.vmap(lambda x: run_episodes(eval_rng, x, max_episode_steps))(gathered_params)
            
            # initial runner state for scanning
            update_steps = 0
            update_runner_state = (train_state, env_state, obsv, partner_indices, rng, update_steps)
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_ep_return_infos)

            init_done = jnp.zeros((config["NUM_CONTROLLED_ACTORS"]), dtype=bool)
            update_runner_state = (
                train_state,
                env_state,
                obsv,
                init_done,
                init_hstate_0,
                partner_indices,
                rng,
                update_steps
            )
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_ep_return_infos)

            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx, eval_ep_return_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "eval_ep_return_infos": metrics["per_iter_ep_infos"]
            }
            return out

        return train
    # ------------------------------
    # 4) Actually run the FCP training
    # ------------------------------
    # training is vmapped across multiple seeds
    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    # rngs = jax.random.split(rng, 1)
    with jax.disable_jit(False):
        #fcp_train_fn = jax.jit(jax.vmap(make_fcp_train(config, partner_params)))
        fcp_train_fn = jax.jit(make_fcp_train(config, partner_params))
        out = fcp_train_fn(rng)
    return out

def open_ended_training(init_fcp_params, others, config, teammate_train_env, fcp_env):

    train_out = train_adversarial_partners(config, init_fcp_params, teammate_train_env)
    train_partner_ckpts = train_out["checkpoints"]

    # Note: Currently training fcp agent against **all** adversarial partner checkpoints
    # TODO Limit training against the last adversarial checkpoints instead
    fcp_out = train_fcp_agent(config, train_partner_ckpts, fcp_env, init_fcp_params)
    updated_fcp_parameters = fcp_out["checkpoints"]
    updated_fcp_parameters = jax.tree_util.tree_map(lambda x: x[-1], updated_fcp_parameters)

    return updated_fcp_parameters, (train_out, fcp_out)

def initialize_agent(config, base_seed):
    rng = jax.random.PRNGKey(base_seed)
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    # S5 specific parameters
    d_model = config["S5_D_MODEL"]
    ssm_size = config["S5_SSM_SIZE"]
    n_layers = config["S5_N_LAYERS"]
    blocks = config["S5_BLOCKS"]
    block_size = int(ssm_size / blocks)

    Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
    block_size = block_size // 2
    ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vinv = V.conj().T

    ssm_init_fn = init_S5SSM(H=d_model,
                            P=ssm_size,
                            Lambda_re_init=Lambda.real,
                            Lambda_im_init=Lambda.imag,
                            V=V,
                            Vinv=Vinv)
    
    agent0_net =  S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       config=config, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],)
    rng, init_rng = jax.random.split(rng)
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # Assume only controlling 1 confederate
    init_x = (
        # init obs, dones, avail_actions
        jnp.zeros((1, config["NUM_CONTROLLED_ACTORS"], env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, config["NUM_CONTROLLED_ACTORS"])),
        jnp.ones((1, config["NUM_CONTROLLED_ACTORS"], env.action_space(env.agents[0]).n)),
    )
    init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)
    init_params = agent0_net.init(init_rng, init_hstate_0, init_x)
    
    return init_params

def process_metrics(outs_train, outs_fcp):

    all_xp_returns = np.asarray(outs_train["all_ep_infos"])
    all_value_losses = np.asarray(outs_train["metrics"]["value_losses"])
    all_actor_losses = np.asarray(outs_train["metrics"]["pg_losses"])
    all_entropy_losses = np.asarray(outs_train["metrics"]["entropy_losses"])

    all_ego_returns = np.asarray(outs_fcp["eval_ep_return_infos"])
    all_ego_value_losses = np.asarray(outs_fcp["metrics"]["value_loss"])
    all_ego_actor_losses = np.asarray(outs_fcp["metrics"]["actor_loss"])
    all_ego_entropy_losses = np.asarray(outs_fcp["metrics"]["entropy_loss"])

    average_xp_rets_per_iter = np.mean(np.mean(all_xp_returns[:, :, :, :, 0], axis=-1), axis=1)
    average_ego_rets_per_iter = np.mean(np.mean(all_ego_returns[:, :, :, :, 0], axis=-1), axis=-1)

    average_value_losses = np.mean(np.mean(np.mean(all_value_losses, axis=-1), axis=-1), axis=1)
    average_actor_losses = np.mean(np.mean(np.mean(all_actor_losses, axis=-1), axis=-1), axis=1)
    average_entropy_losses = np.mean(np.mean(np.mean(all_entropy_losses, axis=-1), axis=-1), axis=1)

    average_ego_value_losses = np.mean(np.mean(all_ego_value_losses, axis=-1), axis=-1)
    average_ego_actor_losses = np.mean(np.mean(all_ego_actor_losses, axis=-1), axis=-1)
    average_ego_entropy_losses = np.mean(np.mean(all_ego_entropy_losses, axis=-1), axis=-1)

    return average_xp_rets_per_iter, average_ego_rets_per_iter, average_value_losses, average_actor_losses, average_entropy_losses, average_ego_value_losses, average_ego_actor_losses, average_ego_entropy_losses


def run_minimax(config):
    algorithm_config = dict(config["algorithm"])
    logger = Logger(config)

    teammate_train_env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    teammate_train_env = LogWrapper(teammate_train_env)
    fcp_env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    fcp_env = LogWrapper(fcp_env)

    partial_with_config = lambda x, y : open_ended_training(x, y, algorithm_config, teammate_train_env, fcp_env)
    init_params = initialize_agent(algorithm_config, 1000)
    fcp_params, others = partial_with_config(init_params, None)

    final_params, outs = jax.lax.scan(partial_with_config, init_params, length=algorithm_config["NUM_OPEN_ENDED_ITERS"])
    outs_train, outs_fcp = outs
    all_metrics = process_metrics(outs_train, outs_fcp)

    for num_iter in range(all_metrics[0].shape[0]):
        for num_step in range(all_metrics[0].shape[1]):
            logger.log_item("Returns/XP", all_metrics[0][num_iter][num_step], checkpoint=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Returns/EgoLearning", all_metrics[1][num_iter][num_step], checkpoint=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageConfValueLoss", all_metrics[2][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageConfActorLoss", all_metrics[3][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageConfEntropy", all_metrics[4][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageEgoValueLoss", all_metrics[5][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageEgoActorLoss", all_metrics[6][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)
            logger.log_item("Losses/AverageEgoEntropyLoss", all_metrics[7][num_iter][num_step], train_step=num_iter*all_metrics[0].shape[1] + num_step)

            logger.commit()

    
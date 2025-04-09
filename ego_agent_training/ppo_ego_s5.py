'''
Initial draft for a generic ego agent training script with a PPO S5 agent. 
The main issue is that currently, the script assumes that the partner is an 
ActorWithDoubleCritic. We want to generalize this to support any partner type / 
allow the user to pass in a partner function.
'''
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

from envs import make_env
from ppo.ippo import unbatchify, Transition
from common.mlp_actor_critic import ActorCritic, ActorWithDoubleCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.wandb_visualizations import Logger

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_s5_ego_agent(config, partner_params, init_ego_params, env, train_rng):
    '''
    Train an S5 ego agent using the given partner checkpoints and IPPO.
    '''
    n_seeds, m_ckpts = partner_params["Dense_0"]["kernel"].shape[:2]
    num_total_partners = n_seeds * m_ckpts

    # helper to gather the correct slice for each environment
    # from shape (n_seeds, m_ckpts, ...) -> (num_envs, ...)
    def unravel_partner_idx(idx):
        """Given a scalar in [0, n_seeds*m_ckpts), return (seed_idx, ckpt_idx)."""
        seed_idx = jnp.floor_divide(idx, m_ckpts)
        ckpt_idx = jnp.mod(idx, m_ckpts)
        return seed_idx, m_ckpts*jnp.ones_like(ckpt_idx) - 1

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

    # ------------------------------
    # Build the FCP training function
    # ------------------------------
    def make_ppo_train(config, partner_params):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["ROLLOUT_LENGTH"] + config["ROLLOUT_LENGTH"]) // config["NUM_ENVS"]
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
            agent0_net = S5ActorCritic(env.action_space(env.agents[0]).n, 
                                       config=config, 
                                       ssm_init_fn=ssm_init_fn,
                                       fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
                                       ssm_hidden_dim=config["S5_SSM_SIZE"],)
            
            init_params = init_ego_params

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

            # Init envs & partner indices
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # Initialize hidden state for ego actor
            init_hstate_0 = StackedEncoderModel.initialize_carry(config["NUM_CONTROLLED_ACTORS"], ssm_size, n_layers)

            # Each environment picks a partner index in [0, n_seeds*m_ckpts)
            rng, partner_rng = jax.random.split(rng)
            partner_indices = jax.random.randint(
                key=partner_rng,
                shape=(config["NUM_UNCONTROLLED_ACTORS"],),
                minval=0,
                maxval=num_total_partners
            )

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
                    pi, _, _ = ActorWithDoubleCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, o)
                    return pi.sample(seed=rng_)

                rng_partner = jax.random.split(partner_rng, config["NUM_UNCONTROLLED_ACTORS"])
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

            # GAE & update step
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
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
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
                metric["actor_loss"] = all_losses[1][1]
                metric["value_loss"] = all_losses[1][0]
                metric["entropy_loss"] = all_losses[1][2]
                new_runner_state = (train_state, env_state, obs, init_done, init_hstate_0, new_partner_idx, rng, update_steps + 1)
                return (new_runner_state, metric)

            # 3e) PPO Update and Checkpoint saving
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def run_single_episode(rng, fcp_param, partner_param):
                # Reset the env.
                rng, reset_rng = jax.random.split(rng)
                obs, env_state = env.reset(reset_rng)
                init_done = jnp.zeros(1, dtype=bool)
                init_returns = jnp.zeros(1, dtype=float)
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
                    init_done.reshape(1, 1), 
                    avail_actions_0
                )

                hstate_0, pi_0, _ = agent0_net.apply(fcp_param, init_hstate_0, rnn_input_0)
                act_0 = pi_0.sample(seed=act_rng).squeeze()

                def apply_partner(p, o, rng_):
                    # p: single-partner param dictionary
                    # o: single obs vector
                    # rng_: single environment's RNG
                    partner_pol, _, _ = ActorWithDoubleCritic(env.action_space(env.agents[1]).n,
                                        activation=config["ACTIVATION"]).apply({'params': p}, o)
                    return partner_pol.sample(seed=rng_)
                
                partner_input = (obs_1, avail_actions_1)
                act1 = apply_partner(partner_param, partner_input, part_rng)    
                both_actions = [act_0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                _, _, eval_rewards, done, dummy_info = env.step(step_rng, env_state, env_act)

                init_returns = init_returns + eval_rewards["agent_0"]

                # We'll use a scan to iterate steps until the episode is done.
                ep_ts = 1
                init_carry = (ep_ts, env_state, obs, rng, done, hstate_0, dummy_info, init_returns)
                def scan_step(carry, _):
                    def take_step(carry_step):
                        ep_ts, env_state, obs, rng, done, hstate_0, last_info, last_total_returns = carry_step
                        # Get available actions for agent 0 from environment state
                        avail_actions = env.get_avail_actions(env_state.env_state)
                        avail_actions = jax.lax.stop_gradient(avail_actions)
                        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                        # Get agent obses
                        obs_0 = obs["agent_0"]
                        obs_1 = obs["agent_1"]
                        prev_done_0 = done["agent_0"]

                        rnn_input_0 = (
                            obs_0.reshape(1, 1, -1),
                            prev_done_0.reshape(1, 1), 
                            avail_actions_0
                        )
                        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                        hstate_0, pi_0, _ = agent0_net.apply(fcp_param, hstate_0, rnn_input_0)
                        act_0 = pi_0.sample(seed=act_rng).squeeze()

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
                        obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)
                        last_total_returns = last_total_returns + reward["agent_0"]

                        return (ep_ts + 1, env_state_next, obs_next, rng, done_next, hstate_0, info_next, last_total_returns)
                            
                    ep_ts, env_state, obs, rng, done, hstate_0, last_info, last_returns = carry
                    new_carry = jax.lax.cond(
                        done["__all__"],
                        # if done, execute true function(operand). else, execute false function(operand).
                        lambda curr_carry: curr_carry, # True fn
                        take_step, # False fn
                        operand=carry
                    )
                    return new_carry, None

                final_carry, _ = jax.lax.scan(
                    scan_step, init_carry, None, length=max_episode_steps)
                # Return the final info (which includes the episode return via LogWrapper).
                return final_carry[-2], final_carry[-1]
            
            def run_episodes(rng, fcp_param, partner_param, num_eps):
                def body_fn(carry, _):
                    rng = carry
                    rng, ep_rng = jax.random.split(rng)
                    ep_info, final_returns = run_single_episode(ep_rng, fcp_param, partner_param)
                    return rng, (ep_info, final_returns)
                rng, all_outs = jax.lax.scan(body_fn, rng, None, length=num_eps)
                return all_outs  # each leaf has shape (num_eps, ...)
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                 checkpoint_array, ckpt_idx, init_eval_info) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)
                

                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = gather_partner_params(partner_params, eval_partner_indices)
                    
                    rng, eval_rng = jax.random.split(rng)
                    eval_ep_return_infos = jax.vmap(lambda x: run_episodes(
                        eval_rng, train_state.params, x, config["MAX_EVAL_EPISODES"]))(gathered_params)
                    return (new_ckpt_arr, cidx + 1, rng, eval_ep_return_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, rng, ep_ret_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, rng, init_eval_info)
                )

                metric["per_iter_ep_infos"] = ep_ret_infos[1]
                return ((train_state, env_state, last_obs, last_done, hstate_0, partner_idx, rng, update_steps),
                        checkpoint_array, ckpt_idx, ep_ret_infos), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # init rngs
            rng, rng_eval, rng_train = jax.random.split(rng, 3)
            # Init eval
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = gather_partner_params(partner_params, eval_partner_indices)
            eval_ep_return_infos = jax.vmap(lambda x: run_episodes(rng_eval, train_state.params, x, config["MAX_EVAL_EPISODES"]))(gathered_params)

            # initial runner state for scanning
            update_steps = 0
            init_done = jnp.zeros((config["NUM_CONTROLLED_ACTORS"]), dtype=bool)
            update_runner_state = (
                train_state,
                env_state,
                obsv,
                init_done,
                init_hstate_0,
                partner_indices,
                rng_train,
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
    # Actually run the PPO training
    # ------------------------------
    out = make_ppo_train(config, partner_params)(train_rng)

    return out

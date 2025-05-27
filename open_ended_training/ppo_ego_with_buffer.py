'''
Script for training a PPO ego agent against a buffered population of partner agents 
that can 
'''
import logging

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.population_interface import AgentPopulation
from agents.population_buffer import BufferedPopulation
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from common.run_episodes import run_episodes

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ppo_ego_agent_with_buffer(config, env, train_rng, 
                           ego_policy, init_ego_params, n_ego_train_seeds,
                           partner_population: AgentPopulation, 
                           population_buffer: BufferedPopulation
                           ):
    '''
    Train PPO ego agent using partners from the BufferedPopulation.

    Args:
        config: dict, config for the training
        env: environment
        train_rng: jax.random.PRNGKey, random key for training
        ego_policy: AgentPolicy, policy for the ego agent
        init_ego_params: dict, initial parameters for the ego agent
        n_ego_train_seeds: int, number of ego training seeds
        partner_population: BufferedPopulation, population manager for partner agents
        population_buffer: PopulationBuffer, buffer containing partner agents
    '''
    # ------------------------------
    # Build the PPO training function
    # ------------------------------
    def make_ppo_train(config):
        '''agent 0 is the ego agent while agent 1 is the confederate'''
        num_agents = env.num_agents
        assert num_agents == 2, "This snippet assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"] # assumption: we control 1 agent
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]

        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0, "NUM_CONTROLLED_ACTORS must be divisible by NUM_MINIBATCHES"
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"], "NUM_CONTROLLED_ACTORS must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
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
                apply_fn=ego_policy.network.apply,
                params=init_ego_params,
                tx=tx,
            )

            # Init envs & partner indices
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # Use the buffered population to sample partner indices for each environment
            rng, partner_rng = jax.random.split(rng)
            partner_indices, current_buffer = partner_population.sample_agent_indices(
                population_buffer, config["NUM_UNCONTROLLED_ACTORS"], partner_rng
            )
            
            def _env_step(runner_state, unused):
                """
                One step of the environment:
                1. Get observations, sample actions from all agents
                2. Step environment using sampled actions
                3. Return state, reward, ...
                """
                train_state, env_state, prev_obs, prev_done, hstate_0, partner_hstate, population_buffer, partner_indices, rng = runner_state
                rng, actor_rng, partner_rng, partner_sample_rng, step_rng = jax.random.split(rng, 5)

                obs_0 = prev_obs["agent_0"]
                obs_1 = prev_obs["agent_1"]

                 # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                
                # Reshape inputs for S5 (sequence_length, batch_size, features)
                obs_0_reshaped = obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                done_0_reshaped = prev_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                
                # Agent_0 (ego) action, value, log_prob
                act_0, val_0, pi_0, hstate_0 = ego_policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=actor_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (partner) action using the BufferedPopulation interface
                # Conditionally resample partners based on prev_done["__all__"]                
                needs_resample = prev_done["__all__"] # shape (NUM_ENVS,) bool

                # Sample potential new indices for all envs & get buffer with updated ages
                # Pass the needs_resample mask to correctly update ages
                sampled_indices_all, updated_buffer = partner_population.sample_agent_indices(
                    population_buffer, # Buffer from previous step
                    config["NUM_UNCONTROLLED_ACTORS"], # Static n = num_envs
                    partner_sample_rng,
                    needs_resample_mask=needs_resample
                )

                # Determine final indices based on whether resampling was needed for each env
                updated_partner_indices = jnp.where(
                    needs_resample,         # Mask shape (NUM_ENVS,)
                    sampled_indices_all,    # Use newly sampled index if True
                    partner_indices         # Else, keep index from previous step
                )
                
                # reshape inputs as necessary for partner
                obs_1_reshaped = obs_1.reshape(config["NUM_UNCONTROLLED_ACTORS"], 1, -1)
                done_1_reshaped = prev_done["agent_1"].reshape(config["NUM_UNCONTROLLED_ACTORS"], 1, -1)

                # Get actions using the final indices and the updated buffer
                act_1, new_partner_hstate = partner_population.get_actions(
                    updated_buffer,    # Use buffer with correct ages
                    updated_partner_indices,  # Use the final selected indices
                    obs_1_reshaped,
                    done_1_reshaped,
                    avail_actions_1,
                    partner_hstate,
                    partner_rng,
                    env_state=env_state,
                    aux_obs=None
                )
                act_1 = act_1.squeeze()

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
                info_0 = jax.tree.map(lambda x: x[:, 0], info) # Assuming agent_0 is the first agent

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
                new_runner_state = (train_state, env_state_next, obs_next, done, 
                                    hstate_0, new_partner_hstate, updated_buffer, 
                                    updated_partner_indices, rng)
                return new_runner_state, transition

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

            def _update_minbatch(train_state, batch_info):
                init_hstate_0, traj_batch, advantages, returns = batch_info
                def _loss_fn(params, init_hstate_0, traj_batch, gae, target_v):
                    _, value, pi, _ = ego_policy.get_action_value_policy(
                        params=params, 
                        obs=traj_batch.obs,
                        done=traj_batch.done,
                        avail_actions=traj_batch.avail_actions,
                        hstate=init_hstate_0,
                        rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                        ).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - target_v)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

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

                # compute average grad norm
                grad_l2_norms = jax.tree.map(lambda g: jnp.linalg.norm(g.astype(jnp.float32)), grads)
                sum_of_grad_norms = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms)
                n_elements = len(jax.tree.leaves(grad_l2_norms))
                avg_grad_norm = sum_of_grad_norms / n_elements
                
                return train_state, (loss_val, aux_vals, avg_grad_norm)

            def _update_epoch(update_state, unused):
                train_state, init_hstate_0, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate_0, config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)
                train_state, losses_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate_0, traj_batch, advantages, targets, rng)
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollouts
                2. Compute advantage
                3. PPO updates
                """
                (train_state, env_state, last_obs, partner_indices, buffer, rng, update_steps) = update_runner_state
                init_hstate_0 = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_partner_hstate = partner_population.init_hstate(config["NUM_UNCONTROLLED_ACTORS"])
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

                # 1) rollout
                runner_state = (train_state, env_state, last_obs, init_done, init_hstate_0, init_partner_hstate, buffer, partner_indices, rng)
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"])
                (train_state, env_state, last_obs, last_done, last_hstate_0, partner_hstate, buffer, partner_indices, rng) = runner_state

                # 2) advantage
                last_obs_batch_0 = last_obs["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
                
                # Reshape inputs for S5
                obs_0_reshaped = last_obs_batch_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1)
                done_0_reshaped = last_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"])
                
                # Get final value estimate for completed trajectory
                _, last_val, _, last_hstate_0 = ego_policy.get_action_value_policy(
                    params=train_state.params, 
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_hstate_0,
                    rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
                )
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
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]
                _, loss_terms, avg_grad_norm = losses_and_grads

                # Reset environment due to update
                rng, reset_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                
                # Metrics
                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric["actor_loss"] = loss_terms[1]
                metric["value_loss"] = loss_terms[0]
                metric["entropy_loss"] = loss_terms[2]
                metric["avg_grad_norm"] = avg_grad_norm
                new_runner_state = (train_state, env_state, obs, partner_indices, buffer, rng, update_steps + 1)
                return (new_runner_state, metric)

            # 3e) PPO Update and Checkpoint saving
            checkpoint_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1) # -1 because we store the final ckpt as the last ckpt
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all FCP checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)

            max_episode_steps = config["ROLLOUT_LENGTH"]
            
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((train_state, env_state, last_obs, partner_idx, buffer, rng, update_steps),
                 checkpoint_array, ckpt_idx, init_eval_last_info) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state, env_state, last_obs, partner_idx, buffer, rng, update_steps),
                    None
                )
                (train_state, env_state, last_obs, partner_idx, buffer, rng, update_steps) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0), 
                                        jnp.equal(update_steps, config["NUM_UPDATES"] - 1))

                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr, train_state.params
                    )
                    rng, eval_rng, partner_rng = jax.random.split(rng, 3)
                    
                    # Sample partners from buffer for evaluation
                    partner_indices, eval_buffer = partner_population.sample_agent_indices(
                        buffer, config["NUM_EVAL_EPISODES"], partner_rng
                    )
                    
                    # Gather parameters for sampled partners
                    gathered_params = partner_population.gather_agent_params(
                        eval_buffer, partner_indices
                    )
                    
                    # Run evaluation with sampled partners
                    eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        eval_rng, env, agent_0_param=train_state.params, agent_0_policy=ego_policy, 
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls, 
                        max_episode_steps=max_episode_steps, 
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params)
                    return (new_ckpt_arr, cidx + 1, rng, eval_eps_last_infos)
                
                def skip_ckpt(args):
                    return args

                (checkpoint_array, ckpt_idx, rng, eval_last_infos) = jax.lax.cond(
                    to_store, store_and_eval_ckpt, skip_ckpt, (checkpoint_array, ckpt_idx, rng, init_eval_last_info)
                )
                # Add evaluation info to metrics
                metric["eval_ep_last_info"] = eval_last_infos

                return ((train_state, env_state, last_obs, partner_idx, buffer, rng, update_steps),
                        checkpoint_array, ckpt_idx, eval_last_infos), metric

            # init checkpoint array
            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0

            # Init eval return infos
            rng, rng_eval, partner_rng, rng_train = jax.random.split(rng, 4)
            
            # Sample partners for initial evaluation
            eval_partner_indices, eval_buffer = partner_population.sample_agent_indices(
                current_buffer, config["NUM_EVAL_EPISODES"], partner_rng
            )
            
            # Gather parameters for partners
            gathered_params = partner_population.gather_agent_params(
                eval_buffer, eval_partner_indices
            )
            
            eval_eps_last_infos = jax.vmap(lambda x: run_episodes(
                        rng_eval, env, 
                        agent_0_param=train_state.params, agent_0_policy=ego_policy, 
                        agent_1_param=x, agent_1_policy=partner_population.policy_cls, 
                        max_episode_steps=max_episode_steps, 
                        num_eps=config["NUM_EVAL_EPISODES"]))(gathered_params)

            # initial runner state for scanning
            update_steps = 0
            update_runner_state = (
                train_state,
                env_state,
                obsv,
                partner_indices,
                current_buffer,
                rng_train,
                update_steps
            )
            state_with_ckpt = (update_runner_state, checkpoint_array, ckpt_idx, eval_eps_last_infos)
            
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (final_runner_state, checkpoint_array, final_ckpt_idx, eval_eps_last_infos) = state_with_ckpt
            out = {
                "final_params": final_runner_state[0].params,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
                "checkpoints": checkpoint_array,
                "final_buffer": final_runner_state[4],  # Extract the final population buffer state
            }
            return out
        return train

    # ------------------------------
    # Actually run the PPO training
    # ------------------------------
    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    train_fn = jax.jit(jax.vmap(make_ppo_train(config)))
    out = train_fn(rngs)    
    return out 
import shutil
import time
import logging
from typing import NamedTuple
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from functools import partial
import copy

from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, S5ActorCriticPolicy
from agents.population_interface import AgentPopulation
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from marl.ppo_utils import Transition, unbatchify, _create_minibatches
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from ego_agent_training.ppo_ego import train_ppo_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ResetTransition(NamedTuple):
    '''Stores extra information for resetting the BR from ego agent states.'''
    env_state: LogEnvState
    conf_obs: jnp.ndarray
    ego_obs: jnp.ndarray
    conf_done: jnp.ndarray
    ego_done: jnp.ndarray
    conf_hstate: jnp.ndarray
    ego_hstate: jnp.ndarray

def train_lagrange_partners(config, env, 
                            ego_params, ego_policy, 
                            conf_params, conf_policy, 
                            br_params, br_policy, partner_rng):
    '''
    Train confederate/best-response pairs that optimize regret within a certain range 
    using the given ego agent policy and IPPO.
    Return model checkpoints and metrics. 
    '''
    def make_lagrange_partner_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        # Define different minibatch sizes for interactions with ego agent and one with BR agent
        config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

        # Right now assume control of just 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_AGENTS"] = config["NUM_ENVS"]

        config["NUM_UPDATES"] = config["TIMESTEPS_PER_ITER_PARTNER"] // (config["ROLLOUT_LENGTH"] * 2 * config["NUM_ENVS"] * config["PARTNER_POP_SIZE"])
        config["MINIBATCH_SIZE"] = config["ROLLOUT_LENGTH"] * config["NUM_CONTROLLED_ACTORS"]

        assert config["MINIBATCH_SIZE"] % config["NUM_MINIBATCHES"] == 0, "MINIBATCH_SIZE must be divisible by NUM_MINIBATCHES"
        assert config["MINIBATCH_SIZE"] >= config["NUM_MINIBATCHES"], "MINIBATCH_SIZE must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng, init_params_conf, init_params_br):
            confederate_policy = conf_policy # ActorWithDoubleCriticPolicy

            upper_lagrange_multiplier = jnp.zeros((1,), dtype=jnp.float32)
            lower_lagrange_multiplier = jnp.zeros((1,), dtype=jnp.float32)
            
            # Define optimizers for both confederate and BR policy
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], 
                eps=1e-5),
            )
            tx_br = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
            )
            train_state_conf = TrainState.create(
                apply_fn=confederate_policy.network.apply,
                params=init_params_conf,
                tx=tx,
            )

            train_state_br = TrainState.create(
                apply_fn=br_policy.network.apply,
                params=init_params_br,
                tx=tx_br,
            )

            # --------------------------
            # 3b) Init envs and hidden states
            # --------------------------
            rng, reset_rng_ego, reset_rng_br = jax.random.split(rng, 3)
            reset_rngs_ego = jax.random.split(reset_rng_ego, config["NUM_ENVS"])
            reset_rngs_br = jax.random.split(reset_rng_br, config["NUM_ENVS"])

            obsv_ego, env_state_ego = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_ego)
            obsv_br, env_state_br = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_br)
            # --------------------------
            # 3c) Define env step
            # --------------------------
            def _env_step_ego(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = ego
                Returns updated runner_state, a Transition for agent_0, 
                and a ResetTransition for resetting to env states encountered here.
                """
                train_state_conf, env_state, last_obs, last_dones, last_conf_h, last_ego_h, rng = runner_state
                rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action using policy interface
                act_0, (val_0, _), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # confederate has same done status
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=act_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent_1 (ego) action using policy interface
                act_1, _, _, new_ego_h = ego_policy.get_action_value_policy(
                    params=ego_params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_ego_h,
                    rng=partner_rng
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
                info_0 = jax.tree.map(lambda x: x[:, 0], info)

                # Store agent_0 data in transition
                transition = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_1"], # we don't negate the ego reward because we're optimizing the lagrange dual objective
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )

                reset_transition = ResetTransition(
                    # all of these are from before env step
                    env_state=env_state,
                    conf_obs=obs_0,
                    ego_obs=obs_1,
                    conf_done=last_dones["agent_0"],
                    ego_done=last_dones["agent_1"],
                    conf_hstate=last_conf_h,
                    ego_hstate=last_ego_h
                )

                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, rng)
                return new_runner_state, (transition, reset_transition)
            
            def _env_step_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, and a Transition for agent_0 and agent_1.
                """
                train_state_conf, train_state_br, env_state, last_obs, last_dones, \
                    last_conf_h, last_br_h, reset_traj_batch, rng = runner_state
                rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)
                
                def gather_sampled(data_pytree, flat_indices, first_nonbatch_dim: int):
                    '''Will treat all dimensions up to the first_nonbatch_dim as batch dimensions. '''
                    batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    flat_data = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[first_nonbatch_dim:]), data_pytree)
                    sampled_data = jax.tree.map(lambda x: x[flat_indices], flat_data) # Shape (N, ...)
                    return sampled_data

                # Reset conf-br data collection from conf-ego states
                if config["RESET_CONF_BR_TO_EGO_STATES"]:

                    rng, sample_rng = jax.random.split(rng)
                    needs_resample = last_dones["__all__"] # shape (N,) bool

                    total_reset_states = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    sampled_indices = jax.random.randint(sample_rng, shape=(config["NUM_ENVS"],), minval=0, 
                                                         maxval=total_reset_states)
                    
                    # Gather sampled leaves from each data pytree
                    sampled_env_state = gather_sampled(reset_traj_batch.env_state, sampled_indices, first_nonbatch_dim=2)
                    sampled_conf_obs = gather_sampled(reset_traj_batch.conf_obs, sampled_indices, first_nonbatch_dim=2)
                    sampled_br_obs = gather_sampled(reset_traj_batch.ego_obs, sampled_indices, first_nonbatch_dim=2)
                    sampled_conf_done = gather_sampled(reset_traj_batch.conf_done, sampled_indices, first_nonbatch_dim=2)
                    sampled_br_done = gather_sampled(reset_traj_batch.ego_done, sampled_indices, first_nonbatch_dim=2)
                    
                    # for done environments, select data corresponding to the reset_traj_batch states
                    env_state = jax.tree.map(
                        lambda sampled, original: jnp.where(
                            needs_resample.reshape((-1,) + (1,) * (original.ndim - 1)), 
                            sampled, original
                        ),
                        sampled_env_state, 
                        env_state
                    )
                    obs_0 = jnp.where(needs_resample[:, jnp.newaxis], sampled_conf_obs, last_obs["agent_0"])
                    obs_1 = jnp.where(needs_resample[:, jnp.newaxis], sampled_br_obs, last_obs["agent_1"])

                    dones_0 = jnp.where(needs_resample, sampled_conf_done, last_dones["agent_0"])
                    dones_1 = jnp.where(needs_resample, sampled_br_done, last_dones["agent_1"])

                    if last_conf_h is not None: # has a leading (1, ) dimension
                         sampled_conf_hstate = gather_sampled(reset_traj_batch.conf_hstate, sampled_indices, first_nonbatch_dim=3)
                         sampled_conf_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sampled_conf_hstate)
                         last_conf_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                        sampled, original), sampled_conf_hstate, last_conf_h)
                    
                    if last_br_h is not None: # has a leading (1, ) dimension
                        if config["REINIT_BR_TO_EGO"]:   
                            sample_br_hstate = gather_sampled(reset_traj_batch.ego_hstate, sampled_indices, first_nonbatch_dim=3)                     
                            sample_br_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sample_br_hstate)
                        else:
                            sample_br_hstate = init_br_hstate # Use the initial state passed in

                        last_br_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                     sampled, original), sample_br_hstate, last_br_h)

                else: # Original logic if not resetting
                    obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                    dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action
                act_0, (_, val_0), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_0.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=conf_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_0 = val_0.squeeze()

                # Agent 1 (best response) action
                act_1, val_1, pi_1, new_br_h = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_1.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1),
                    hstate=last_br_h,
                    rng=br_rng
                )
                logp_1 = pi_1.log_prob(act_1)

                act_1 = act_1.squeeze()
                logp_1 = logp_1.squeeze()
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
                info_0 = jax.tree.map(lambda x: x[:, 0], info)
                info_1 = jax.tree.map(lambda x: x[:, 1], info)

                # Store agent_0 (confederate) data in transition
                transition_0 = Transition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_0,
                    reward=reward["agent_0"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                # Store agent_1 (best response) data in transition
                transition_1 = Transition(
                    done=done["agent_1"],
                    action=act_1,
                    value=val_1,
                    reward=reward["agent_1"],
                    log_prob=logp_1,
                    obs=obs_1,
                    info=info_1,
                    avail_actions=avail_actions_1
                )
                # Pass reset_traj_batch and init_br_hstate through unchanged in the state tuple
                new_runner_state = (train_state_conf, train_state_br, env_state_next, obs_next, done, new_conf_h, new_br_h, reset_traj_batch, rng)
                return new_runner_state, (transition_0, transition_1)
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
                def _update_minbatch_conf(train_state_conf, batch_infos):
                    minbatch_ego, minbatch_br, lower_lm, upper_lm = batch_infos
                    init_conf_hstate_ego, traj_batch_ego, advantages_ego, returns_ego = minbatch_ego
                    init_conf_hstate_br, traj_batch_br, advantages_br, returns_br = minbatch_br

                    def _loss_fn_conf(params, traj_batch_ego, gae_ego, target_v_ego, traj_batch_br, gae_br, target_v_br):
                        # get policy and value of confederate versus ego and best response agents respectively

                        _, (value_ego_conf_ego_data, value_br_conf_ego_data), pi_ego, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_ego.obs, 
                            done=traj_batch_ego.done,
                            avail_actions=traj_batch_ego.avail_actions,
                            hstate=init_conf_hstate_ego,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        _, (value_ego_conf_br_data, value_br_conf_br_data), pi_br, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_br.obs, 
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_conf_hstate_br,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_ego = pi_ego.log_prob(traj_batch_ego.action)
                        log_prob_br = pi_br.log_prob(traj_batch_br.action)

                        # Value loss for interaction with ego agent
                        value_pred_ego_clipped = traj_batch_ego.value + (
                            value_ego_conf_ego_data - traj_batch_ego.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_ego = jnp.square(value_ego_conf_ego_data - target_v_ego)
                        value_losses_clipped_ego = jnp.square(value_pred_ego_clipped - target_v_ego)
                        value_loss_ego = (
                            jnp.maximum(value_losses_ego, value_losses_clipped_ego).mean()
                        )

                        # Value loss for interaction with best response agent
                        value_pred_br_clipped = traj_batch_br.value + (
                            value_br_conf_br_data - traj_batch_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_br = jnp.square(value_br_conf_br_data - target_v_br)
                        value_losses_clipped_br = jnp.square(value_pred_br_clipped - target_v_br)
                        value_loss_br = (
                            jnp.maximum(value_losses_br, value_losses_clipped_br).mean()
                        )

                        # Policy gradient loss for interaction with ego agent
                        ratio_ego = jnp.exp(log_prob_ego - traj_batch_ego.log_prob)
                        ratio_br = jnp.exp(log_prob_br - traj_batch_br.log_prob)
                        
                        # Compute policy gradient objectives
                        if config["CONF_OBJ_TYPE"] == "per_state_regret":
                            # Compute return-to-gos
                            conf_br_return_to_go_br_data = value_br_conf_br_data + advantages_br
                            conf_ego_return_to_go_ego_data = value_ego_conf_ego_data + advantages_ego

                            # Compute regret-related terms
                            regret_ego_data = value_br_conf_ego_data - conf_ego_return_to_go_ego_data
                            regret_br_data = conf_br_return_to_go_br_data - value_ego_conf_br_data

                            upper_lagrange_dual_diff_ego = config["UPPER_REGRET_THRESHOLD"] - regret_ego_data
                            lower_lagrange_dual_diff_ego = regret_ego_data - config["LOWER_REGRET_THRESHOLD"]
                            upper_lagrange_dual_diff_br = config["UPPER_REGRET_THRESHOLD"] - regret_br_data
                            lower_lagrange_dual_diff_br = regret_br_data - config["LOWER_REGRET_THRESHOLD"]

                            # total_br_objective = conf_br_return_to_go_br_data + upper_lm * upper_lagrange_dual_diff_br + lower_lm * lower_lagrange_dual_diff_br
                            # total_ego_objective = value_br_conf_ego_data + upper_lm * upper_lagrange_dual_diff_ego + lower_lm * lower_lagrange_dual_diff_ego

                            total_br_objective = gae_br + upper_lm * upper_lagrange_dual_diff_br + lower_lm * lower_lagrange_dual_diff_br
                            total_ego_objective = gae_ego + upper_lm * upper_lagrange_dual_diff_ego + lower_lm * lower_lagrange_dual_diff_ego
                        
                        elif config["CONF_OBJ_TYPE"] == "traj_level_regret":
                            total_br_objective = gae_br * (1 + lower_lm - upper_lm)
                            total_ego_objective = gae_ego * (upper_lm - lower_lm)

                        # Policy gradient loss for interaction with best response agent
                        normalized_total_br_objective = (total_br_objective - total_br_objective.mean()) / (total_br_objective.std() + 1e-8)
                        normalized_total_ego_objective = (total_ego_objective - total_ego_objective.mean()) / (total_ego_objective.std() + 1e-8)

                        pg_loss_1_br = ratio_br * normalized_total_br_objective
                        pg_loss_2_br = jnp.clip(
                            ratio_br, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * normalized_total_br_objective
                        pg_loss_br = -jnp.mean(jnp.minimum(pg_loss_1_br, pg_loss_2_br))

                        pg_loss_1_ego_opt = ratio_ego * normalized_total_ego_objective
                        pg_loss_2_ego_opt = jnp.clip(
                            ratio_ego, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * normalized_total_ego_objective
                        pg_loss_ego_opt = -jnp.mean(jnp.minimum(pg_loss_1_ego_opt, pg_loss_2_ego_opt))

                        # Entropy for interaction with ego agent and best response agents resp.
                        entropy_ego = jnp.mean(pi_ego.entropy())
                        entropy_br = jnp.mean(pi_br.entropy())
                        
                        total_weight = 1 + upper_lm + lower_lm
                        
                        loss_ego =  pg_loss_ego_opt  + config["VF_COEF"] * value_loss_ego -  total_weight * config["ENT_COEF"] * entropy_ego
                        loss_br = pg_loss_br + config["VF_COEF"] * value_loss_br - total_weight * config["ENT_COEF"] * entropy_br
                        total_loss = loss_ego + loss_br
                        return total_loss, (value_loss_ego, value_loss_br, pg_loss_ego_opt, pg_loss_br, entropy_ego, entropy_br)
                    
                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params, 
                        traj_batch_ego, advantages_ego, returns_ego, 
                        traj_batch_br, advantages_br, returns_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)
                
                def _update_minbatch_br(train_state_br, batch_info):
                    batch_info_br, lower_lm, upper_lm = batch_info
                    init_br_hstate, traj_batch_br, advantages, returns = batch_info_br
                    
                    def _loss_fn_br(params, traj_batch_br, gae, target_v):
                        _, value, pi, _ = br_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_br.obs, 
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_br_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        log_prob = pi.log_prob(traj_batch_br.action)

                        # Value loss
                        value_pred_clipped = traj_batch_br.value + (
                            value - traj_batch_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - target_v)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss
                        # sp_weight = 1 + lower_lm - upper_lm
                        ratio = jnp.exp(log_prob - traj_batch_br.log_prob)
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

                    grad_fn = jax.value_and_grad(_loss_fn_br, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_br.params, traj_batch_br, advantages, returns)
                    train_state_br = train_state_br.apply_gradients(grads=grads)
                    return train_state_br, (loss_val, aux_vals)

                def _update_lagrange(conf_train_state, minibatches_ego, 
                                     minibatches_0_br, lower_lm, upper_lm):
                    
                    init_conf_hstate_ego, traj_batches1, _, targets_batch_1 = minibatches_ego
                    init_conf_hstate_br, traj_batches2, _, targets_batch_2 = minibatches_0_br
                    
                    _, (value_ego1, value_br1), _, _ = confederate_policy.get_action_value_policy(
                            params=conf_train_state.params, 
                            obs=traj_batches1.obs, 
                            done=traj_batches1.done,
                            avail_actions=traj_batches1.avail_actions,
                            hstate=init_conf_hstate_ego,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                    
                    _, (value_ego2, value_br2), _, _ = confederate_policy.get_action_value_policy(
                            params=conf_train_state.params, 
                            obs=traj_batches2.obs, 
                            done=traj_batches2.done,
                            avail_actions=traj_batches2.avail_actions,
                            hstate=init_conf_hstate_br,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                    
                    if config["LAGRANGE_USE_TARGET_RETURNS"]:
                        combined_ego = jnp.concatenate([targets_batch_1, value_ego2], axis=0)
                        combined_value_br = jnp.concatenate([value_br1, targets_batch_2], axis=0)
                    else:
                        combined_ego = jnp.concatenate([value_ego1, value_ego2], axis=0)
                        combined_value_br = jnp.concatenate([value_br1, value_br2], axis=0)

                    lower_diff = combined_value_br - combined_ego - config["LOWER_REGRET_THRESHOLD"]
                    upper_diff = combined_ego + config["UPPER_REGRET_THRESHOLD"] - combined_value_br

                    lower_diff_mean, upper_diff_mean = lower_diff.mean(), upper_diff.mean()
                    new_lower_lm = lower_lm - (config["LAGRANGE_MULTIPLIER_LR"] * lower_diff_mean) # TODO: try using an optimizer for this
                    new_upper_lm = upper_lm - (config["LAGRANGE_MULTIPLIER_LR"] * upper_diff_mean)

                    new_lower_lm = jnp.maximum(new_lower_lm, jnp.zeros_like(new_lower_lm))
                    new_upper_lm = jnp.maximum(new_upper_lm, jnp.zeros_like(new_upper_lm))

                    return jnp.reshape(new_lower_lm, (1,)), jnp.reshape(new_upper_lm, (1,)), lower_diff_mean, upper_diff_mean
                
                (
                    train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br, 
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br, lower_lm, upper_lm
                ) = update_state

                rng_ego, perm_rng_ego, perm_rng_br0, perm_rng_br1 = jax.random.split(rng_ego, 4)

                # Create minibatches for each agent and interaction type
                init_conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_conf_hstate_br = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                init_hstate_br = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

                minibatches_ego = _create_minibatches(
                    traj_batch_ego, advantages_conf_ego, targets_conf_ego, init_conf_hstate_ego, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_ego
                )
                minibatches_br0 = _create_minibatches(
                    traj_batch_br_0, advantages_conf_br, targets_conf_br, init_conf_hstate_br, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br0
                )
                minibatches_br1 = _create_minibatches(
                    traj_batch_br_1, advantages_br, targets_br, init_hstate_br, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_br1
                )

                old_train_state_conf, old_train_state_br = train_state_conf, train_state_br
                repeated_lower_lm = jnp.tile(lower_lm, minibatches_ego[1].obs.shape[0])
                repeated_upper_lm = jnp.tile(upper_lm, minibatches_ego[1].obs.shape[0])

                # Update confederate
                train_state_conf, total_loss = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, 
                    (minibatches_ego, minibatches_br0, repeated_lower_lm, repeated_upper_lm)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, (minibatches_br1, repeated_lower_lm, repeated_upper_lm)
                )

                lower_lm, upper_lm, lower_diff_mean, upper_diff_mean = _update_lagrange(
                    old_train_state_conf, minibatches_ego, minibatches_br0, lower_lm, upper_lm
                )
                lm_grads = (lower_diff_mean, upper_diff_mean)
                update_state = (train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br,
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br, lower_lm, upper_lm)
                return update_state, (total_loss, total_loss_br, lm_grads)

            def _update_step(update_runner_state, unused):
                """
                1. Collect rollout for interactions against ego agent.
                2. Collect rollout for interactions against br agent.
                3. Compute advantages for ego-conf and conf-br interactions.
                4. PPO updates for best response and confederate policies.
                """
                (
                    train_state_conf, train_state_br, 
                    env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, 
                    last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, 
                    conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps,
                    lower_lm, upper_lm
                ) = update_runner_state

                # 1) rollout for interactions against ego agent
                runner_state_ego = (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                                    conf_hstate_ego, ego_hstate, rng_ego)
                runner_state_ego, (traj_batch_ego, reset_traj_batch_ego) = jax.lax.scan(
                    _env_step_ego, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                 conf_hstate_ego, ego_hstate, rng_ego) = runner_state_ego

                # 2) rollout for interactions against br agent
                runner_state_br = (train_state_conf, train_state_br, env_state_br, last_obs_br, 
                                   last_dones_br, conf_hstate_br, br_hstate, reset_traj_batch_ego, rng_br)
                runner_state_br, (traj_batch_br_0, traj_batch_br_1 ) = jax.lax.scan(
                    _env_step_br, runner_state_br, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_br, last_obs_br, last_dones_br, 
                conf_hstate_br, br_hstate, _,  rng_br) = runner_state_br

                # 3a) compute advantage for confederate agent from interaction with ego agent

                # Get available actions for agent 0 from environment state
                avail_actions_0_ego = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_0"].astype(jnp.float32)
                
                # Get last value
                _, (last_val_0_ego, _), _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1), 
                    done=last_dones_ego["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_ego),
                    hstate=conf_hstate_ego,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_ego = last_val_0_ego.squeeze()
                advantages_conf_ego, targets_conf_ego = _calculate_gae(traj_batch_ego, last_val_0_ego)

                # 3b) compute advantage for confederate agent from interaction with br policy

                # Get available actions for agent 0 from environment state
                avail_actions_0_br = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_0"].astype(jnp.float32)
                
                # Get last value using agent interface
                _, (_, last_val_0_br), _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_br["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_br["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0_br),
                    hstate=conf_hstate_br,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_0_br = last_val_0_br.squeeze()
                advantages_conf_br, targets_conf_br = _calculate_gae(traj_batch_br_0, last_val_0_br)

                # 3c) compute advantage for br policy from interaction with confederate agent

                avail_actions_1_br = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_1"].astype(jnp.float32)
                # Get last value using agent interface
                _, last_val_1_br, _, _ = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=last_obs_br["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones_br["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_1_br),
                    hstate=br_hstate,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                last_val_1_br = last_val_1_br.squeeze()
                advantages_br, targets_br = _calculate_gae(traj_batch_br_1, last_val_1_br)

                # 3) PPO update
                update_state = (
                    train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br,
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br, lower_lm, upper_lm
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]
                lower_lm = update_state[-2]
                upper_lm = update_state[-1]

                # Metrics
                metric = traj_batch_ego.info
                metric["update_steps"] = update_steps
                metric["upper_lm"] = upper_lm
                metric["lower_lm"] = lower_lm
                metric["value_loss_conf_against_ego"] = all_losses[0][1][0]
                metric["value_loss_conf_against_br"] = all_losses[0][1][1]
                metric["pg_loss_conf_against_ego"] = all_losses[0][1][2]
                metric["pg_loss_conf_against_br"] = all_losses[0][1][3]
                metric["entropy_conf_against_ego"] = all_losses[0][1][4]
                metric["entropy_conf_against_br"] = all_losses[0][1][5]
                metric["average_rewards_br"] = jnp.mean(traj_batch_br_0.reward)
                metric["average_rewards_ego"] = jnp.mean(traj_batch_ego.reward)
                metric["lm_lower_diff_mean"] = all_losses[2][0]
                metric["lm_upper_diff_mean"] = all_losses[2][1]

                #value_loss_br, pg_loss_br, entropy_loss_br = total_loss_br[1]
                metric["value_loss_br"] = all_losses[1][1][0]
                metric["pg_loss_br"] = all_losses[1][1][1]
                metric["entropy_loss_br"] = all_losses[1][1][2]

                new_runner_state = (
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,  
                    rng_ego, rng_br, update_steps + 1, lower_lm, upper_lm
                )
                return (new_runner_state, metric)

            # --------------------------
            # PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1) # -1 because we store the final ckpt as the last ckpt
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)
        
            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br , last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps, lower_lm, upper_lm
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps, lower_lm, upper_lm),
                    None
                )

                (
                    train_state_conf, train_state_br, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps, lower_lm, upper_lm
                ) = new_runner_state

                # Decide if we store a checkpoint
                to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0), 
                                          jnp.equal(update_steps, config["NUM_UPDATES"] - 1))
                
                def store_and_eval_ckpt(args):
                    ckpt_arr_and_ep_infos, rng, cidx = args
                    ckpt_arr_conf, ckpt_arr_br, prev_ep_infos_br, prev_ep_infos_ego = ckpt_arr_and_ep_infos
                    new_ckpt_arr_conf = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_conf, train_state_conf.params
                    )
                    new_ckpt_arr_br = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr_br, train_state_br.params
                    )

                    # run eval episodes
                    rng, eval_rng, = jax.random.split(rng)
                    # conf vs ego
                    last_ep_info_with_ego = run_episodes(eval_rng, env, 
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(eval_rng, env, 
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=train_state_conf.params, agent_1_policy=confederate_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    
                    return ((new_ckpt_arr_conf, new_ckpt_arr_br, last_ep_info_with_br, last_ep_info_with_ego), rng, cidx + 1)

                def skip_ckpt(args):
                    return args

                (checkpoint_array_and_infos, rng_ego, ckpt_idx) = jax.lax.cond(
                    to_store, 
                    store_and_eval_ckpt, 
                    skip_ckpt, 
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info_br, eval_info_ego), rng_ego, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, ep_info_br, ep_info_ego = checkpoint_array_and_infos
                
                metric["eval_ep_last_info_br"] = ep_info_br
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return ((train_state_conf, train_state_br, env_state_ego, env_state_br, 
                         last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                         conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                         rng_ego, rng_br, update_steps, lower_lm, upper_lm),
                         checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                         ep_info_br, ep_info_ego), metric

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_br = init_ckpt_array(train_state_br.params)
            ckpt_idx = 0

            # initial state for scan over _update_step_with_ckpt
            update_steps = 0
            rng, rng_eval_ego, rng_eval_br = jax.random.split(rng, 3)
            ep_infos_ego = run_episodes(rng_eval_ego, env, 
                agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
            )
            ep_infos_br = run_episodes(rng_eval_br, env, 
                agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                agent_1_param=ego_params, agent_1_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"])

            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_br = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])


            # Initialize done flags
            init_dones_ego = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_br = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

            rng, rng_ego, rng_br = jax.random.split(rng, 3)
            update_runner_state = (
                train_state_conf, train_state_br, env_state_ego, env_state_br, 
                obsv_ego, obsv_br, init_dones_ego, init_dones_br, 
                init_conf_hstate_ego, init_ego_hstate, init_conf_hstate_br, init_br_hstate, 
                rng_ego, rng_br, update_steps, lower_lagrange_multiplier, upper_lagrange_multiplier
            )
            state_with_ckpt = (
                update_runner_state, checkpoint_array_conf, checkpoint_array_br, 
                ckpt_idx, ep_infos_br, ep_infos_ego
            )
            # run training
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"]
            )
            (
                final_runner_state, checkpoint_array_conf, checkpoint_array_br, 
                final_ckpt_idx, last_ep_infos_br, last_ep_infos_ego
            ) = state_with_ckpt

            out = {
                "final_params_conf": final_runner_state[0].params,
                "final_params_br": final_runner_state[1].params,
                "checkpoints_conf": checkpoint_array_conf,
                "checkpoints_br": checkpoint_array_br,
                "metrics": metrics,  # shape (NUM_UPDATES, ...)
            }
            return out

        return train
    # ------------------------------
    # Actually run the adversarial teammate training
    # ------------------------------
    rngs = jax.random.split(partner_rng, config["PARTNER_POP_SIZE"])
    train_fn = jax.jit(jax.vmap(make_lagrange_partner_train(config)))
    out = train_fn(rngs, conf_params, br_params)
    return out

def linear_schedule_regret(iter_idx, config):
    '''Computes the upper and lower regret thresholds based on the iteration index. 
    Updates the config with the next regret thresholds.'''
    frac = iter_idx / config["NUM_OPEN_ENDED_ITERS"]
    config["LOWER_REGRET_THRESHOLD"] = config["LOWER_REGRET_THRESHOLD_START"] + (config["LOWER_REGRET_THRESHOLD_END"] - config["LOWER_REGRET_THRESHOLD_START"]) * frac
    config["UPPER_REGRET_THRESHOLD"] = config["UPPER_REGRET_THRESHOLD_START"] + (config["UPPER_REGRET_THRESHOLD_END"] - config["UPPER_REGRET_THRESHOLD_START"]) * frac
    return config

def open_ended_training_step(carry, ego_policy, conf_policy, br_policy, partner_population, oe_config, ego_config, env):
    '''
    Train the ego agent against the regret-maximizing partners. 
    '''
    prev_ego_params, prev_conf_params, prev_br_params, rng, oel_iter_idx = carry
    rng, partner_rng, ego_rng, conf_init_rng, br_init_rng = jax.random.split(rng, 5)
    
    oe_config = linear_schedule_regret(oel_iter_idx, oe_config) # update regret thresholds
    
    if oe_config["REINIT_CONF"]:
        init_rngs = jax.random.split(conf_init_rng, oe_config["PARTNER_POP_SIZE"])
        conf_params = jax.vmap(conf_policy.init_params)(init_rngs)
    else:
        conf_params = prev_conf_params

    if oe_config["REINIT_BR_TO_BR"]:
        init_rngs = jax.random.split(br_init_rng, oe_config["PARTNER_POP_SIZE"])
        br_params = jax.vmap(br_policy.init_params)(init_rngs)
    elif oe_config["REINIT_BR_TO_EGO"]:
        br_params = jax.tree.map(lambda x: x[jnp.newaxis, ...].repeat(oe_config["PARTNER_POP_SIZE"], axis=0), prev_ego_params)
    else:
        br_params = prev_br_params
    
    # Train partner agents with ego_policy
    train_out = train_lagrange_partners(oe_config, env,
                                        ego_params=prev_ego_params, ego_policy=ego_policy,
                                        conf_params=conf_params, conf_policy=conf_policy, 
                                        br_params=br_params, br_policy=br_policy, 
                                        partner_rng=partner_rng)
    
    if oe_config["EGO_TEAMMATE"] == "final":
        train_partner_params = train_out["final_params_conf"]

    elif oe_config["EGO_TEAMMATE"] == "all":
        n_ckpts = oe_config["PARTNER_POP_SIZE"] * oe_config["NUM_CHECKPOINTS"]
        train_partner_params = jax.tree.map(
            lambda x: x.reshape((n_ckpts,) + x.shape[2:]), 
            train_out["checkpoints_conf"]
        )
    
    # Train ego agent using train_ppo_ego_agent
    ego_out = train_ppo_ego_agent(
        config=ego_config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=train_partner_params
    )
    
    updated_ego_parameters = ego_out["final_params"]
    updated_conf_parameters = train_out["final_params_conf"]
    updated_br_parameters = train_out["final_params_br"]

    # remove initial dimension of 1, to ensure that input and output ego parameters have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_conf_parameters, updated_br_parameters, rng, oel_iter_idx + 1)
    return carry, (train_out, ego_out)

def train_lagrange(rng, env, algorithm_config, ego_config):
    rng, init_ego_rng, init_conf_rng, init_br_rng, train_rng = jax.random.split(rng, 5)
    
    # Initialize ego agent
    ego_policy, init_ego_params = initialize_s5_agent(ego_config, env, init_ego_rng)
    
    # initialize PARTNER_POP_SIZE conf and br params
    conf_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
    )
    init_conf_rngs = jax.random.split(init_conf_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_conf_params = jax.vmap(conf_policy.init_params)(init_conf_rngs)

    assert not (algorithm_config["REINIT_BR_TO_EGO"] and algorithm_config["REINIT_BR_TO_BR"]), "Cannot reinitialize br to both ego and br"
    if algorithm_config["REINIT_BR_TO_EGO"]:
        # initialize br policy to have same architecture as ego policy
        # a bit hacky
        br_policy = S5ActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            d_model=algorithm_config.get("S5_D_MODEL", 16),
            ssm_size=algorithm_config.get("S5_SSM_SIZE", 16),
            n_layers=algorithm_config.get("S5_N_LAYERS", 2),
            blocks=algorithm_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=algorithm_config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
            s5_activation=algorithm_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=algorithm_config.get("S5_DO_NORM", True),
            s5_prenorm=algorithm_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=algorithm_config.get("S5_DO_GTRXL_NORM", True),
        )
    else:
        br_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
        )

    init_br_rngs = jax.random.split(init_br_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_br_params = jax.vmap(br_policy.init_params)(init_br_rngs)

    # Create partner population
    if algorithm_config["EGO_TEAMMATE"] == "final":
        pop_size = algorithm_config["PARTNER_POP_SIZE"]
    elif algorithm_config["EGO_TEAMMATE"] == "all":
        pop_size = algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"]
    else:
        raise ValueError(f"Invalid value for EGO_TEAMMATE: {algorithm_config['EGO_TEAMMATE']}")
    
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=conf_policy
    )
    
    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                        partner_population, algorithm_config, ego_config, env)
    
    init_carry = (init_ego_params, init_conf_params, init_br_params, train_rng, 0)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    return outs

def run_lagrange(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    # initialize ego config
    ego_config = copy.deepcopy(algorithm_config)
    ego_config["TOTAL_TIMESTEPS"] = algorithm_config["TIMESTEPS_PER_ITER_EGO"]
    EGO_ARGS = algorithm_config.get("EGO_ARGS", {})
    ego_config.update(EGO_ARGS)
    
    log.info("Starting open-ended Lagrange training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_lagrange, 
                env=env, algorithm_config=algorithm_config, ego_config=ego_config
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Open-ended Lagrange training completed in {end_time - start_time} seconds.")
    
    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    return ego_policy, ego_params, init_ego_params


def log_metrics(config, logger, outs, metric_names: tuple):
    """Process training metrics and log them using the provided logger.
    
    Args:
        config: dict, the configuration
        outs: tuple, contains (teammate_outs, ego_outs) for each iteration
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    teammate_outs, ego_outs = outs
    teammate_metrics = teammate_outs["metrics"] # conf vs ego 
    ego_metrics = ego_outs["metrics"]

    num_open_ended_iters = ego_metrics["returned_episode_returns"].shape[1]
    num_partner_updates = teammate_metrics["returned_episode_returns"].shape[3]
    num_ego_updates = ego_metrics["returned_episode_returns"].shape[3]

    # Extract partner train stats
    teammate_metrics = jax.tree.map(lambda x: x, teammate_metrics)
    teammate_stats = get_stats(teammate_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, 2)
    teammate_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], teammate_stats) # shape (num_open_ended_iters, num_partner_updates)

    # Extract ego train stats
    ego_stats = get_stats(ego_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, 2)
    ego_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], ego_stats) # shape (num_open_ended_iters, num_ego_updates)

    # Process/extract PAIRED-specific losses    
    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    avg_teammate_sp_returns = np.asarray(teammate_metrics["eval_ep_last_info_br"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5))
    avg_teammate_xp_returns = np.asarray(teammate_metrics["eval_ep_last_info_ego"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5))
    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, update_epochs)
    avg_lm_lower_diff_mean = np.asarray(teammate_metrics["lm_lower_diff_mean"]).mean(axis=(0, 2, 4))
    avg_lm_upper_diff_mean = np.asarray(teammate_metrics["lm_upper_diff_mean"]).mean(axis=(0, 2, 4))

    # Conf vs ego, conf vs br, br losses
    #  shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, update_epochs, num_minibatches)
    avg_value_losses_teammate_against_ego = np.asarray(teammate_metrics["value_loss_conf_against_ego"]).mean(axis=(0, 2, 4, 5))
    avg_value_losses_teammate_against_br = np.asarray(teammate_metrics["value_loss_conf_against_br"]).mean(axis=(0, 2, 4, 5)) 
    avg_value_losses_br = np.asarray(teammate_metrics["value_loss_br"]).mean(axis=(0, 2, 4, 5))
    
    avg_actor_losses_teammate_against_ego = np.asarray(teammate_metrics["pg_loss_conf_against_ego"]).mean(axis=(0, 2, 4, 5)) 
    avg_actor_losses_teammate_against_br = np.asarray(teammate_metrics["pg_loss_conf_against_br"]).mean(axis=(0, 2, 4, 5))
    avg_actor_losses_br = np.asarray(teammate_metrics["pg_loss_br"]).mean(axis=(0, 2, 4, 5))
    
    avg_entropy_losses_teammate_against_ego = np.asarray(teammate_metrics["entropy_conf_against_ego"]).mean(axis=(0, 2, 4, 5))
    avg_entropy_losses_teammate_against_br = np.asarray(teammate_metrics["entropy_conf_against_br"]).mean(axis=(0, 2, 4, 5))
    avg_entropy_losses_br = np.asarray(teammate_metrics["entropy_loss_br"]).mean(axis=(0, 2, 4, 5))
    
    avg_lagrange_upper = np.asarray(teammate_metrics["upper_lm"]).mean(axis=(0, 2, 4))
    avg_lagrange_lower = np.asarray(teammate_metrics["lower_lm"]).mean(axis=(0, 2, 4))

    # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates)
    avg_rewards_teammate_against_br = np.asarray(teammate_metrics["average_rewards_br"]).mean(axis=(0, 2))
    avg_rewards_teammate_against_ego = np.asarray(teammate_metrics["average_rewards_ego"]).mean(axis=(0, 2))
    
    # Process ego-specific metrics
    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, num_partners, num_eval_episodes, num_agents_per_env)
    avg_ego_returns = np.asarray(ego_metrics["eval_ep_last_info"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5, 6))
    # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, update_epochs, num_minibatches)
    avg_ego_value_losses = np.asarray(ego_metrics["value_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_actor_losses = np.asarray(ego_metrics["actor_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_entropy_losses = np.asarray(ego_metrics["entropy_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_grad_norms = np.asarray(ego_metrics["avg_grad_norm"]).mean(axis=(0, 2, 4, 5))

    for iter_idx in range(num_open_ended_iters):        
        # Log all partner metrics
        for step in range(num_partner_updates):
            global_step = iter_idx * num_partner_updates + step
            
            # Log standard partner stats from get_stats
            for stat_name, stat_data in teammate_stat_means.items():
                logger.log_item(f"Train/Conf-Against-Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)
            
            # Log paired-specific metrics
            # Eval metrics
            logger.log_item("Eval/ConfReturn-Against-Ego", avg_teammate_xp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/ConfReturn-Against-BR", avg_teammate_sp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/EgoRegret", avg_teammate_sp_returns[iter_idx][step] - avg_teammate_xp_returns[iter_idx][step], train_step=global_step)
            # Confederate losses
            logger.log_item("Losses/ConfValLoss-Against-Ego", avg_value_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfActorLoss-Against-Ego", avg_actor_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfEntropy-Against-Ego", avg_entropy_losses_teammate_against_ego[iter_idx][step], train_step=global_step)
            
            logger.log_item("Losses/ConfValLoss-Against-BR", avg_value_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfActorLoss-Against-BR", avg_actor_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/ConfEntropy-Against-BR", avg_entropy_losses_teammate_against_br[iter_idx][step], train_step=global_step)
            
            # Best response losses
            logger.log_item("Losses/BRValLoss", avg_value_losses_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/BRActorLoss", avg_actor_losses_br[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/BREntropyLoss", avg_entropy_losses_br[iter_idx][step], train_step=global_step)

            # Lagrange learning
            logger.log_item("Losses/LowerLagrangeMagnitude", avg_lagrange_lower[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/UpperLagrangeMagnitude", avg_lagrange_upper[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/LagrangeLowerDiffMean", avg_lm_lower_diff_mean[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/LagrangeUpperDiffMean", avg_lm_upper_diff_mean[iter_idx][step], train_step=global_step)
        
            # Rewards
            logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/AvgConfBRRewards", avg_rewards_teammate_against_br[iter_idx][step], train_step=global_step)

        ### Ego metrics processing
        for step in range(num_ego_updates):
            global_step = iter_idx * num_ego_updates + step
            # Standard ego stats from get_stats
            for stat_name, stat_data in ego_stat_means.items():
                logger.log_item(f"Train/Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)

            # Ego eval metrics
            logger.log_item("Eval/EgoReturn-Against-Conf", avg_ego_returns[iter_idx][step], train_step=global_step)

            # Ego agent losses
            logger.log_item("Losses/EgoValueLoss", avg_ego_value_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoActorLoss", avg_ego_actor_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoEntropyLoss", avg_ego_entropy_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoGradNorm", avg_ego_grad_norms[iter_idx][step], train_step=global_step)
    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
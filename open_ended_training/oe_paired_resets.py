'''Open ended PAIRED, resetting conf-ego to conf-br states and vice versa. '''
import time
import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.population_interface import AgentPopulation
from agents.initialize_agents import initialize_s5_agent
from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, S5ActorCriticPolicy
from common.plot_utils import get_metric_names
from common.run_episodes import run_episodes
from common.ppo_utils import Transition, unbatchify
from envs import make_env
from envs.log_wrapper import LogWrapper, LogEnvState
from ego_agent_training.ppo_ego import train_ppo_ego_agent
from open_ended_training.open_ended_paired import log_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ResetTransition(NamedTuple):
    '''Stores extra information for resetting agents to a point in some trajectory.'''
    env_state: LogEnvState
    conf_obs: jnp.ndarray
    partner_obs: jnp.ndarray
    conf_done: jnp.ndarray
    partner_done: jnp.ndarray
    conf_hstate: jnp.ndarray
    partner_hstate: jnp.ndarray

class ConfTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    other_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def _create_minibatches(traj_batch, advantages, targets, init_hstate, num_actors, num_minibatches, perm_rng):
    """Create minibatches for PPO updates, where each leaf has shape 
        (num_minibatches, rollout_len, num_actors / num_minibatches, ...) 
    This function creates minibatches shuffled only over the num_actors = num_envs * num_agents_per_env
    dimension, thus ensuring that the rollout (time) dimension is preserved. 
    This makes the minibatches compatible with recurrent ActorCritics.
    """
    # Create batch containing trajectory, advantages, and targets

    batch = (
        init_hstate, # shape (1, num_actors, hidden_dim) = (1, 16, 64)
        traj_batch, # pytree: obs is shape (rollout_len, num_actors, feat_shape) = (128, 16, 15) 
        advantages, # shape (rollout_len, num_actors) = (128, 16)
        targets # shape (rollout_len, num_actors) = (128, 16)
            )

    permutation = jax.random.permutation(perm_rng, num_actors)

    # each leaf of shuffled batch has shape (rollout_len, num_actors=num_envs*num_agents_per_env, feat_shape)
    # except for init_hstate which has shape (1, num_actors=num_envs*num_agents_per_env, hidden_dim)
    shuffled_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    # each leaf has shape (num_minibatches, rollout_len, num_actors/num_minibatches, feat_shape)
    # except for init_hstate which has shape (num_minibatches, 1, num_actors/num_minibatches, hidden_dim)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1] 
                + list(x.shape[2:]),
        ), 1, 0,),
        shuffled_batch,
    )
    return minibatches

def train_regret_maximizing_partners(config, env, 
                                     ego_params, ego_policy, 
                                     conf_params, conf_policy, 
                                     br_params, br_policy, partner_rng):
    '''
    Train regret-maximizing confederate/best-response pairs using the given ego agent policy and IPPO.
    Return model checkpoints and metrics. 
    '''
    def make_regret_maximizing_partner_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

        # Define different minibatch sizes for interactions with ego agent and one with BR agent
        config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

        # Right now assume control of just 1 agent
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_AGENTS"] = config["NUM_ENVS"]

        # TODO: enable using a different number of envs for each of the 4 types of interactions.
        # Dividing by 3 for now for simplicity. 
        config["NUM_UPDATES"] = config["TIMESTEPS_PER_ITER_PARTNER"] // (config["ROLLOUT_LENGTH"] * 3 * config["NUM_ENVS"] * config["PARTNER_POP_SIZE"])
        config["MINIBATCH_SIZE"] = config["ROLLOUT_LENGTH"] * config["NUM_CONTROLLED_ACTORS"]

        assert config["MINIBATCH_SIZE"] % config["NUM_MINIBATCHES"] == 0, "MINIBATCH_SIZE must be divisible by NUM_MINIBATCHES"
        assert config["MINIBATCH_SIZE"] >= config["NUM_MINIBATCHES"], "MINIBATCH_SIZE must be >= NUM_MINIBATCHES"

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng, init_params_conf, init_params_br):
            confederate_policy = conf_policy
            
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

            def _reset_to_states(reset_traj_batch, env_state, last_obs, last_dones, last_conf_h, last_partner_h, 
                                 init_partner_hstate, rng, partner_is_br: bool):
                '''Resets the env_state and the hstates to the values in reset_traj_batch for done environments.'''

                def gather_sampled(data_pytree, flat_indices, first_nonbatch_dim: int):
                    '''Will treat all dimensions up to the first_nonbatch_dim as batch dimensions. '''
                    batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                    flat_data = jax.tree.map(lambda x: x.reshape(batch_size, *x.shape[first_nonbatch_dim:]), data_pytree)
                    sampled_data = jax.tree.map(lambda x: x[flat_indices], flat_data) # Shape (N, ...)
                    return sampled_data
                
                rng, sample_rng = jax.random.split(rng)
                needs_resample = last_dones["__all__"] # shape (N,) bool

                total_reset_states = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
                sampled_indices = jax.random.randint(sample_rng, shape=(config["NUM_ENVS"],), minval=0, 
                                                        maxval=total_reset_states)
                
                # Gather sampled leaves from each data pytree
                sampled_env_state = gather_sampled(reset_traj_batch.env_state, sampled_indices, first_nonbatch_dim=2)
                sampled_conf_obs = gather_sampled(reset_traj_batch.conf_obs, sampled_indices, first_nonbatch_dim=2)
                sampled_partner_obs = gather_sampled(reset_traj_batch.partner_obs, sampled_indices, first_nonbatch_dim=2)
                sampled_conf_done = gather_sampled(reset_traj_batch.conf_done, sampled_indices, first_nonbatch_dim=2)
                sampled_partner_done = gather_sampled(reset_traj_batch.partner_done, sampled_indices, first_nonbatch_dim=2)
                
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
                obs_1 = jnp.where(needs_resample[:, jnp.newaxis], sampled_partner_obs, last_obs["agent_1"])

                dones_0 = jnp.where(needs_resample, sampled_conf_done, last_dones["agent_0"])
                dones_1 = jnp.where(needs_resample, sampled_partner_done, last_dones["agent_1"])

                if last_conf_h is not None: # has a leading (1, ) dimension
                    sampled_conf_hstate = gather_sampled(reset_traj_batch.conf_hstate, sampled_indices, first_nonbatch_dim=3)
                    sampled_conf_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sampled_conf_hstate)
                    last_conf_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                   sampled, original), sampled_conf_hstate, last_conf_h)
                
                if last_partner_h is not None: # has a leading (1, ) dimension
                    if config["REINIT_BR_TO_EGO"] and partner_is_br: # we know br is compatible with the ego hstate
                        sample_partner_hstate = gather_sampled(reset_traj_batch.partner_hstate, sampled_indices, first_nonbatch_dim=3)                     
                        sample_partner_hstate = jax.tree.map(lambda x: x[jnp.newaxis, ...], sample_partner_hstate)
                    else: # otherwise, just reset to the provided partner initial hstate
                        sample_partner_hstate = init_partner_hstate # Use the initial state passed in

                    last_partner_h = jax.tree.map(lambda sampled, original: jnp.where(needs_resample[jnp.newaxis, :, jnp.newaxis], 
                                                                                    sampled, original), sample_partner_hstate, last_partner_h)
                return env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_partner_h
            
            def _env_step_conf_ego(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = ego
                Returns updated runner_state, a ConfTransition for the confederate, and a ResetTransition.
                """
                train_state_conf, env_state, last_obs, last_dones, last_conf_h, last_ego_h, reset_traj_batch, rng = runner_state
                rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                # Reset conf-ego data collection from conf-br states
                if reset_traj_batch is not None:
                    env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_ego_h = _reset_to_states(
                        reset_traj_batch, env_state, last_obs, last_dones, 
                        last_conf_h, last_ego_h, init_ego_hstate, rng, partner_is_br=False)
                else: # Original logic if not resetting
                    obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                    dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action using policy interface
                act_0, (val_ego, val_br), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_0.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=jax.lax.stop_gradient(avail_actions_0),
                    hstate=last_conf_h,
                    rng=act_rng
                )
                logp_0 = pi_0.log_prob(act_0)

                act_0 = act_0.squeeze()
                logp_0 = logp_0.squeeze()
                val_ego = val_ego.squeeze()
                val_br = val_br.squeeze()

                # Agent_1 (ego) action using policy interface
                act_1, _, _, new_ego_h = ego_policy.get_action_value_policy(
                    params=ego_params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=dones_1.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
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
                transition = ConfTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_ego,
                    other_value=val_br,
                    reward=reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                reset_transition = ResetTransition(
                    # all of these are from before env step
                    env_state=env_state,
                    conf_obs=obs_0,
                    partner_obs=obs_1,
                    conf_done=last_dones["agent_0"],
                    partner_done=last_dones["agent_1"],
                    conf_hstate=last_conf_h,
                    partner_hstate=last_ego_h
                )

                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, reset_traj_batch, rng)
                return new_runner_state, (transition, reset_transition)

            def _env_step_conf_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, ConfTransition for the confederate, 
                Transition for the best response, and a ResetTransition.
                """
                train_state_conf, train_state_br, env_state, last_obs, last_dones, \
                    last_conf_h, last_br_h, reset_traj_batch, rng = runner_state
                rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)
                
                # Reset conf-br data collection from conf-ego states
                if reset_traj_batch is not None:
                    env_state, obs_0, obs_1, dones_0, dones_1, last_conf_h, last_br_h = _reset_to_states(
                        reset_traj_batch, env_state, last_obs, last_dones, 
                        last_conf_h, last_br_h, init_br_hstate, rng, partner_is_br=True)
                else: # Original logic if not resetting
                    obs_0, obs_1 = last_obs["agent_0"], last_obs["agent_1"]
                    dones_0, dones_1 = last_dones["agent_0"], last_dones["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action
                act_0, (val_ego, val_br), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
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
                val_ego = val_ego.squeeze()
                val_br = val_br.squeeze()

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
                transition_0 = ConfTransition(
                    done=done["agent_0"],
                    action=act_0,
                    value=val_br,
                    other_value=val_ego,
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
                reset_transition = ResetTransition(
                    # all of these are from before env step
                    env_state=env_state,
                    conf_obs=obs_0,
                    partner_obs=obs_1,
                    conf_done=last_dones["agent_0"],
                    partner_done=last_dones["agent_1"],
                    conf_hstate=last_conf_h,
                    partner_hstate=last_br_h
                )
                
                # Pass reset_traj_batch and init_br_hstate through unchanged in the state tuple
                new_runner_state = (train_state_conf, train_state_br, env_state_next, obs_next, done, new_conf_h, new_br_h, reset_traj_batch, rng)
                return new_runner_state, (transition_0, transition_1, reset_transition)
            
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
                def _compute_ppo_value_loss(pred_value, traj_batch, target_v):
                    '''Value loss function for PPO'''
                    value_pred_clipped = traj_batch.value + (
                        pred_value - traj_batch.value
                        ).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(pred_value - target_v)
                    value_losses_clipped = jnp.square(value_pred_clipped - target_v)
                    value_loss = (
                        jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    return value_loss
            
                def _compute_ppo_pg_loss(objective, log_prob, traj_batch):
                    '''Policy gradient loss function for PPO'''
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    obj_norm = (objective - objective.mean()) / (objective.std() + 1e-8)
                    pg_loss_1 = ratio * obj_norm
                    pg_loss_2 = jnp.clip(
                        ratio, 
                        1.0 - config["CLIP_EPS"], 
                        1.0 + config["CLIP_EPS"]) * obj_norm
                    pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
                    return pg_loss

                def _update_minbatch_conf(train_state_conf, batch_infos):
                    minbatch_xp, minbatch_sp = batch_infos
                    init_conf_hstate, traj_batch_xp, advantages_xp, returns_xp = minbatch_xp
                    init_conf_hstate, traj_batch_sp, advantages_sp, returns_sp = minbatch_sp

                    def _loss_fn_conf(params, traj_batch_xp, gae_xp, target_v_xp, 
                                      traj_batch_sp, gae_sp, target_v_sp):
                        # get policy and value of confederate versus ego and best response agents respectively
                        _, (value_xp_on_xp_data, value_sp_on_xp_data), pi_xp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_xp.obs, 
                            done=traj_batch_xp.done,
                            avail_actions=traj_batch_xp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        _, (value_xp_on_sp_data, value_sp_on_sp_data), pi_sp, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_sp.obs, 
                            done=traj_batch_sp.done,
                            avail_actions=traj_batch_sp.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_xp = pi_xp.log_prob(traj_batch_xp.action)
                        log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)
                        
                        value_loss_xp = _compute_ppo_value_loss(value_xp_on_xp_data, traj_batch_xp, target_v_xp)
                        value_loss_sp = _compute_ppo_value_loss(value_sp_on_sp_data, traj_batch_sp, target_v_sp)

                        # Compute policy gradient objectives
                        if config["CONF_OBJ_TYPE"] == "per_state_regret":
                            xp_return_to_go_xp_data = value_xp_on_xp_data + advantages_xp
                            sp_return_to_go_sp_data = value_sp_on_sp_data + advantages_sp
                            regret_xp_data = config["REGRET_SP_WEIGHT"] * value_sp_on_xp_data - xp_return_to_go_xp_data
                            regret_sp_data = config["REGRET_SP_WEIGHT"] * sp_return_to_go_sp_data - value_xp_on_sp_data

                            total_xp_objective = regret_xp_data
                            total_sp_objective = regret_sp_data

                        elif config["CONF_OBJ_TYPE"] == "per_state_regret_target":
                            # use target returns and values to compute the objectives
                            regret_xp_data = config["REGRET_SP_WEIGHT"] * traj_batch_xp.other_value - target_v_xp
                            regret_sp_data = config["REGRET_SP_WEIGHT"] * target_v_sp - traj_batch_sp.other_value

                            total_xp_objective = regret_xp_data
                            total_sp_objective = regret_sp_data

                        elif config["CONF_OBJ_TYPE"] == "traj_level_regret":
                            total_xp_objective = -gae_xp
                            total_sp_objective = gae_sp

                        pg_loss_xp = _compute_ppo_pg_loss(total_xp_objective, log_prob_xp, traj_batch_xp)
                        pg_loss_sp = _compute_ppo_pg_loss(total_sp_objective, log_prob_sp, traj_batch_sp)

                        entropy_xp = jnp.mean(pi_xp.entropy())
                        entropy_sp = jnp.mean(pi_sp.entropy())

                        xp_loss = pg_loss_xp + config["VF_COEF"] * value_loss_xp - config["ENT_COEF"] * entropy_xp
                        sp_loss = pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp

                        total_loss = config["CONF_BR_WEIGHT"] * sp_loss + xp_loss
                        return total_loss, (value_loss_xp, value_loss_sp, 
                                            pg_loss_xp, pg_loss_sp, 
                                            entropy_xp, entropy_sp)

                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params, 
                        traj_batch_xp, advantages_xp, returns_xp, 
                        traj_batch_sp, advantages_sp, returns_sp)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)
                
                def _update_minbatch_br(train_state_br, batch_infos):
                    minbatch_sp = batch_infos
                    init_br_hstate, traj_batch_sp, advantages_sp, returns_sp = minbatch_sp

                    def _loss_fn_br(params, traj_batch_sp, gae_sp, target_v_sp):
                        _, value_sp, pi_sp, _ = br_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_sp.obs, 
                            done=traj_batch_sp.done,
                            avail_actions=traj_batch_sp.avail_actions,
                            hstate=init_br_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_sp = pi_sp.log_prob(traj_batch_sp.action)

                        value_loss_sp = _compute_ppo_value_loss(value_sp, traj_batch_sp, target_v_sp)
                        pg_loss_sp = _compute_ppo_pg_loss(gae_sp, log_prob_sp, traj_batch_sp)
                        entropy_sp = jnp.mean(pi_sp.entropy())

                        sp_loss = pg_loss_sp + config["VF_COEF"] * value_loss_sp - config["ENT_COEF"] * entropy_sp

                        total_loss = sp_loss
                        return total_loss, (value_loss_sp, pg_loss_sp, entropy_sp)

                    # all traj_batch_br keys have shape (128, 4, ...) as does advantages and returns
                    grad_fn = jax.value_and_grad(_loss_fn_br, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_br.params, 
                        traj_batch_sp, advantages_sp, returns_sp)
                    train_state_br = train_state_br.apply_gradients(grads=grads)
                    return train_state_br, (loss_val, aux_vals)

                
                (
                    train_state_conf, train_state_br, 
                    traj_batch_xp, traj_batch_sp_conf, traj_batch_sp_br, 
                    advantages_xp_conf, advantages_sp_conf, advantages_sp_br, 
                    targets_xp_conf, targets_sp_conf, targets_sp_br, 
                    rng
                ) = update_state

                rng, perm_rng_xp, perm_rng_sp_conf, perm_rng_sp_br = jax.random.split(rng, 4)

                # Create minibatches for each agent and interaction type
                minibatches_xp = _create_minibatches(
                    traj_batch_xp, advantages_xp_conf, targets_xp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_xp
                )
                minibatches_sp_conf = _create_minibatches(
                    traj_batch_sp_conf, advantages_sp_conf, targets_sp_conf, init_conf_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sp_conf
                )
                minibatches_sp_br = _create_minibatches(
                    traj_batch_sp_br, advantages_sp_br, targets_sp_br, init_br_hstate, 
                    config["NUM_CONTROLLED_ACTORS"], config["NUM_MINIBATCHES"], perm_rng_sp_br
                )

                # Update confederate
                train_state_conf, total_loss_conf = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_xp, minibatches_sp_conf)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, minibatches_sp_br
                )

                update_state = (train_state_conf, train_state_br, 
                    traj_batch_xp, traj_batch_sp_conf, traj_batch_sp_br, 
                    advantages_xp_conf, advantages_sp_conf, advantages_sp_br, 
                    targets_xp_conf, targets_sp_conf, targets_sp_br, 
                    rng)
                return update_state, (total_loss_conf, total_loss_br)

            def _update_step(update_runner_state, unused):
                """
                1. Collect confederate-ego rollout (XP).
                2. Collect confederate-br rollout (SP).
                3. Collect confederate-ego rollout from confederate-br states (MP1).
                4. Collect confederate-br rollout from confederate-ego states (MP2). 
                5. Compute advantages for XP, and SP interactions.
                6. PPO updates for best response and confederate policies.
                """
                (
                    train_state_conf, train_state_br, 
                    env_state_xp, env_state_sp, env_state_mp1, env_state_mp2, 
                    last_obs_xp, last_obs_sp, last_obs_mp1, last_obs_mp2, 
                    last_dones_xp, last_dones_sp, last_dones_mp1, last_dones_mp2, 
                    conf_hstate_xp, ego_hstate_xp, 
                    conf_hstate_sp, br_hstate_sp, 
                    conf_hstate_mp1, ego_hstate_mp1, 
                    conf_hstate_mp2, br_hstate_mp2, 
                    rng_update, update_steps
                ) = update_runner_state

                rng_update, rng_xp, rng_sp, rng_mp1, rng_mp2 = jax.random.split(rng_update, 5)

                # 1) rollout for conf-ego interaction (XP)
                runner_state_xp = (train_state_conf, env_state_xp, last_obs_xp, last_dones_xp,
                                    conf_hstate_xp, ego_hstate_xp, None, rng_xp)
                runner_state_xp, (traj_batch_xp, reset_traj_batch_xp) = jax.lax.scan(
                    _env_step_conf_ego, runner_state_xp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_xp, last_obs_xp, last_dones_xp, 
                 conf_hstate_xp, ego_hstate_xp, _, rng_xp) = runner_state_xp
            
                # 2) rollout for conf-br interaction (SP)
                runner_state_sp = (train_state_conf, train_state_br, env_state_sp, last_obs_sp, 
                                   last_dones_sp, conf_hstate_sp, br_hstate_sp, None, rng_sp)
                runner_state_sp, (traj_batch_sp_conf, traj_batch_sp_br, reset_traj_batch_sp) = jax.lax.scan(
                    _env_step_conf_br, runner_state_sp, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_sp, last_obs_sp, last_dones_sp,
                conf_hstate_sp, br_hstate_sp, _, rng_sp) = runner_state_sp
                
                # 3) rollout for conf-ego interaction from conf-br states (MP1)
                runner_state_mp1 = (train_state_conf, env_state_mp1, last_obs_mp1, last_dones_mp1,
                                    conf_hstate_mp1, ego_hstate_mp1, 
                                    reset_traj_batch_sp, rng_mp1)
                runner_state_mp1, (traj_batch_mp1, _) = jax.lax.scan(
                    _env_step_conf_ego, runner_state_mp1, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_mp1, last_obs_mp1, last_dones_mp1, 
                 conf_hstate_mp1, ego_hstate_mp1, _, rng_mp1) = runner_state_mp1

                # 4) rollout for conf-br interaction from conf-ego states (MP2)
                runner_state_mp2 = (train_state_conf, train_state_br, env_state_mp2, last_obs_mp2, 
                                    last_dones_mp2, conf_hstate_mp2, br_hstate_mp2, 
                                    reset_traj_batch_xp, rng_mp2)
                runner_state_mp2, (traj_batch_mp2_conf, traj_batch_mp2_br, _) = jax.lax.scan(
                    _env_step_conf_br, runner_state_mp2, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_mp2, last_obs_mp2, last_dones_mp2,
                 conf_hstate_mp2, br_hstate_mp2, _, rng_mp2) = runner_state_mp2

                # Merge XP and MP1 data to compute conf advantages for SP interaction
                # for the traj batch axis 0 is the time dimension, axis 1 is the batch dimension
                traj_batch_xp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), 
                                             traj_batch_xp, traj_batch_mp1)
                # for transition data, axis 0 is the batch dimension
                env_state_xp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 env_state_xp, env_state_mp1)
                last_obs_xp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 last_obs_xp, last_obs_mp1)
                last_dones_xp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 last_dones_xp, last_dones_mp1)
                conf_hstate_xp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 conf_hstate_xp, conf_hstate_mp1)
                
                # Merge confederate SP and MP2 rollout data 
                traj_batch_sp_conf_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), 
                                                 traj_batch_sp_conf, traj_batch_mp2_conf)

                traj_batch_sp_br_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), 
                                            traj_batch_sp_br, traj_batch_mp2_br)

                conf_hstate_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 conf_hstate_sp, conf_hstate_mp2)
                br_hstate_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 br_hstate_sp, br_hstate_mp2)
                
                env_state_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 env_state_sp, env_state_mp2)
                last_obs_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 last_obs_sp, last_obs_mp2)
                last_dones_sp_merged = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), 
                                                 last_dones_sp, last_dones_mp2)
                

                def _compute_advantages_and_targets(batch_size, env_state, policy, policy_params, policy_hstate, 
                                                   last_obs, last_dones, traj_batch, agent_name, value_idx=None):
                    '''Value_idx argument is to support the ActorWithDoubleCritic (confederate) policy, which 
                    has two value heads. Value head 0 models the ego agent while value head 1 models the best response.'''
                    avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)[agent_name].astype(jnp.float32)
                    _, vals, _, _ = policy.get_action_value_policy(
                        params=policy_params,
                        obs=last_obs[agent_name].reshape(1, batch_size, -1),
                        done=last_dones[agent_name].reshape(1, batch_size),
                        avail_actions=jax.lax.stop_gradient(avail_actions),
                        hstate=policy_hstate,
                        rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                    )
                    if value_idx is None:
                        last_val = vals.squeeze()
                    else:
                        last_val = vals[value_idx].squeeze()
                    advantages, targets = _calculate_gae(traj_batch, last_val)
                    return advantages, targets
                
                # 5a) Compute conf advantages for XP (conf-ego) interaction
                advantages_xp_conf, targets_xp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"]*2,
                    env_state_xp_merged, conf_policy, train_state_conf.params, conf_hstate_xp_merged, 
                    last_obs_xp_merged, last_dones_xp_merged, traj_batch_xp_merged, "agent_0", value_idx=0)

                # 5b) Compute conf and br advantages for SP (conf-br) interaction
                advantages_sp_conf, targets_sp_conf = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"]*2,
                    env_state_sp_merged, conf_policy, train_state_conf.params, conf_hstate_sp_merged, 
                    last_obs_sp_merged, last_dones_sp_merged, traj_batch_sp_conf_merged, "agent_0", value_idx=1)
                
                advantages_sp_br, targets_sp_br = _compute_advantages_and_targets(
                    config["NUM_CONTROLLED_ACTORS"]*2,
                    env_state_sp_merged, br_policy, train_state_br.params, br_hstate_sp_merged, 
                    last_obs_sp_merged, last_dones_sp_merged, traj_batch_sp_br_merged, "agent_1", value_idx=None)

                # 6) PPO update
                rng_update, sub_rng = jax.random.split(rng_update, 2)
                update_state = (
                    train_state_conf, train_state_br, 
                    traj_batch_xp_merged, traj_batch_sp_conf_merged, traj_batch_sp_br_merged,
                    advantages_xp_conf, advantages_sp_conf, advantages_sp_br,
                    targets_xp_conf, targets_sp_conf, targets_sp_br,
                    sub_rng
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]

                conf_losses, br_losses = all_losses

                (conf_value_loss_xp, conf_value_loss_sp, 
                 conf_pg_loss_xp, conf_pg_loss_sp, 
                 conf_entropy_xp, conf_entropy_sp) = conf_losses[1]

                (br_value_loss_sp, br_pg_loss_sp, br_entropy_sp) = br_losses[1]
                
                # Metrics
                metric = traj_batch_xp.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_against_ego"] = conf_value_loss_xp
                metric["value_loss_conf_against_br"] = conf_value_loss_sp

                metric["pg_loss_conf_against_ego"] = conf_pg_loss_xp
                metric["pg_loss_conf_against_br"] = conf_pg_loss_sp
                # metric["pg_loss_conf_against_br_mp2"] = conf_pg_loss_mp2

                metric["entropy_conf_against_ego"] = conf_entropy_xp
                metric["entropy_conf_against_br"] = conf_entropy_sp
                # metric["entropy_conf_against_br_mp2"] = conf_entropy_mp2
                
                metric["average_rewards_ego"] = jnp.mean(traj_batch_sp_conf.reward)
                metric["average_rewards_br"] = jnp.mean(traj_batch_sp_br.reward)
                # metric["average_rewards_br_mp2"] = jnp.mean(traj_batch_mp2_br.reward)

                metric["value_loss_br"] = br_value_loss_sp
                # metric["value_loss_br_mp2"] = br_value_loss_mp2

                metric["pg_loss_br"] = br_pg_loss_sp
                # metric["pg_loss_br_mp2"] = br_pg_loss_mp2

                metric["entropy_loss_br"] = br_entropy_sp
                # metric["entropy_loss_br_mp2"] = br_entropy_mp2

                new_update_runner_state = (
                    train_state_conf, train_state_br, 
                    env_state_xp, env_state_sp, env_state_mp1, env_state_mp2, 
                    last_obs_xp, last_obs_sp, last_obs_mp1, last_obs_mp2, 
                    last_dones_xp, last_dones_sp, last_dones_mp1, last_dones_mp2, 
                    conf_hstate_xp, ego_hstate_xp, 
                    conf_hstate_sp, br_hstate_sp, 
                    conf_hstate_mp1, ego_hstate_mp1, 
                    conf_hstate_mp2, br_hstate_mp2, 
                    rng_update, update_steps + 1
                )
                return (new_update_runner_state, metric)

            # --------------------------
            # PPO Update and Checkpoint saving
            # --------------------------
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            # Build a PyTree that holds parameters for all conf agent checkpoints
            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype), 
                    params_pytree)
        
            def _update_step_with_ckpt(state_with_ckpt, unused):
                (update_runner_state, checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_update_runner_state, metric) = _update_step(
                    update_runner_state,
                    None
                )

                rng, update_steps = new_update_runner_state[-2], new_update_runner_state[-1]

                # Decide if we store a checkpoint
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)
                
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
                rng, store_and_eval_rng = jax.random.split(rng, 2)
                (checkpoint_array_and_infos, store_and_eval_rng, ckpt_idx) = jax.lax.cond(
                    to_store, 
                    store_and_eval_ckpt, 
                    skip_ckpt, 
                    ((checkpoint_array_conf, checkpoint_array_br, eval_info_br, eval_info_ego), store_and_eval_rng, ckpt_idx)
                )
                checkpoint_array_conf, checkpoint_array_br, ep_info_br, ep_info_ego = checkpoint_array_and_infos
                
                metric["eval_ep_last_info_br"] = ep_info_br
                metric["eval_ep_last_info_ego"] = ep_info_ego

                return (new_update_runner_state,
                        checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                        ep_info_br, ep_info_ego), metric

            # --------------------------
            # Init all variables for train loop
            # --------------------------
            rng, reset_rng_xp, reset_rng_sp, reset_rng_mp1, reset_rng_mp2 = jax.random.split(rng, 5)
            reset_rngs_xp = jax.random.split(reset_rng_xp, config["NUM_ENVS"])
            reset_rngs_sp = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
            reset_rngs_mp1 = jax.random.split(reset_rng_mp1, config["NUM_ENVS"])
            reset_rngs_mp2 = jax.random.split(reset_rng_mp2, config["NUM_ENVS"])

            obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xp)
            obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sp)
            obsv_mp1, env_state_mp1 = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mp1)
            obsv_mp2, env_state_mp2 = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mp2)

            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])

            # init checkpoint array
            checkpoint_array_conf = init_ckpt_array(train_state_conf.params)
            checkpoint_array_br = init_ckpt_array(train_state_br.params)
            ckpt_idx = 0

            # initial ep_infos for scan over _update_step_with_ckpt
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

            # Initialize done flags
            init_dones_xp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_sp = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_mp1 = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_dones_mp2 = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            
            # Initialize update runner state
            rng, rng_update = jax.random.split(rng, 2)
            update_steps = 0

            update_runner_state = (
                train_state_conf, train_state_br, 
                env_state_xp, env_state_sp, env_state_mp1, env_state_mp2,
                obsv_xp, obsv_sp, obsv_mp1, obsv_mp2, 
                init_dones_xp, init_dones_sp, init_dones_mp1, init_dones_mp2,
                init_conf_hstate, init_ego_hstate, # hstates for conf-ego XP interaction
                init_conf_hstate, init_br_hstate, # hstates for conf-br SP interaction
                init_conf_hstate, init_ego_hstate, # hstates for conf-ego MP1 interaction
                init_conf_hstate, init_br_hstate, # hstates for conf-br MP2 interaction
                rng_update, update_steps
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
    train_fn = jax.jit(jax.vmap(make_regret_maximizing_partner_train(config)))
    out = train_fn(rngs, conf_params, br_params)
    return out

def open_ended_training_step(carry, ego_policy, conf_policy, br_policy, partner_population, config, env):
    '''
    Train the ego agent against the regret-maximizing partners. 
    Note: Currently training fcp agent against **all** adversarial partner checkpoints
    '''
    prev_ego_params, prev_conf_params, prev_br_params, rng = carry
    rng, partner_rng, ego_rng, conf_init_rng, br_init_rng = jax.random.split(rng, 5)
    
    if config["REINIT_CONF"]:
        init_rngs = jax.random.split(conf_init_rng, config["PARTNER_POP_SIZE"])
        conf_params = jax.vmap(conf_policy.init_params)(init_rngs)
    else:
        conf_params = prev_conf_params

    if config["REINIT_BR_TO_BR"]:
        init_rngs = jax.random.split(br_init_rng, config["PARTNER_POP_SIZE"])
        br_params = jax.vmap(br_policy.init_params)(init_rngs)
    elif config["REINIT_BR_TO_EGO"]:
        br_params = jax.tree.map(lambda x: x[jnp.newaxis, ...].repeat(config["PARTNER_POP_SIZE"], axis=0), prev_ego_params)
    else:
        br_params = prev_br_params

    # Train partner agents with ego_policy
    train_out = train_regret_maximizing_partners(config, env, 
                                                 ego_params=prev_ego_params, ego_policy=ego_policy, 
                                                 conf_params=conf_params, conf_policy=conf_policy, 
                                                 br_params=br_params, br_policy=br_policy,
                                                 partner_rng=partner_rng
                                                 )
    train_partner_params = train_out["checkpoints_conf"]
    
    # Reshape partner parameters for AgentPopulation
    pop_size = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]

    # Flatten partner parameters for AgentPopulation
    flattened_partner_params = jax.tree.map(
        lambda x: x.reshape((pop_size,) + x.shape[2:]), 
        train_partner_params
    )
    
    # Train ego agent using train_ppo_ego_agent
    config["TOTAL_TIMESTEPS"] = config["TIMESTEPS_PER_ITER_EGO"]
    ego_out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )
    
    updated_ego_parameters = ego_out["final_params"]
    updated_conf_parameters = train_out["final_params_conf"]
    updated_br_parameters = train_out["final_params_br"]

    # remove initial dimension of 1, to ensure that input and output ego parameters have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_conf_parameters, updated_br_parameters, rng)
    return carry, (train_out, ego_out)


def train_paired(rng, env, algorithm_config):
    rng, init_ego_rng, init_conf_rng, init_br_rng, train_rng = jax.random.split(rng, 5)
    
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

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
    partner_population = AgentPopulation(
        pop_size=algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"],
        policy_cls=conf_policy
    )

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                        partner_population, algorithm_config, env)
    
    init_carry = (init_ego_params, init_conf_params, init_br_params, train_rng)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    return outs

def run_paired(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])


    log.info("Starting open-ended PAIRED w/resets training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_paired, 
                env=env, algorithm_config=algorithm_config
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Open-ended PAIRED w/resets training completed in {end_time - start_time} seconds.")

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    return ego_policy, ego_params, init_ego_params

import os
import time
import logging

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


from agents.agent_interface import AgentPopulation, ActorWithDoubleCriticPolicy, MLPActorCriticPolicy
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.wandb_visualizations import Logger
from common.run_episodes import run_episodes
from envs import make_env
from envs.log_wrapper import LogWrapper
from ego_agent_training.ppo_ego import train_ppo_ego_agent
from ppo.ippo import unbatchify, Transition
from open_ended_training.heldout_eval import run_heldout_evaluation, log_heldout_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_regret_maximizing_partners(config, ego_params, ego_policy, env, partner_rng):
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

        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["ROLLOUT_LENGTH"] + config["ROLLOUT_LENGTH"])// config["NUM_ENVS"]
        config["MINIBATCH_SIZE_EGO"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]
        config["MINIBATCH_SIZE_BR"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            # Initialize confederate policy using ActorWithDoubleCriticPolicy
            confederate_policy = ActorWithDoubleCriticPolicy(
                action_dim=env.action_space(env.agents[0]).n,
                obs_dim=env.observation_space(env.agents[0]).shape[0]
            )
            
            # Initialize best response policy using MLPActorCriticPolicy
            br_policy = MLPActorCriticPolicy(
                action_dim=env.action_space(env.agents[1]).n,
                obs_dim=env.observation_space(env.agents[1]).shape[0]
            )
            
            rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
            
            # Initialize parameters using the policy interfaces
            init_params_conf = confederate_policy.init_params(init_conf_rng)
            init_params_br = br_policy.init_params(init_br_rng)

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
                Returns updated runner_state, and a Transition for agent_0.
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
                    reward=-reward["agent_1"],
                    log_prob=logp_0,
                    obs=obs_0,
                    info=info_0,
                    avail_actions=avail_actions_0
                )
                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_conf_h, new_ego_h, rng)
                return new_runner_state, transition
            
            def _env_step_br(runner_state, unused):
                """
                agent_0 = confederate, agent_1 = best response
                Returns updated runner_state, and a Transition for agent_0 and agent_1.
                """
                train_state_conf, train_state_br, env_state, last_obs, last_dones, last_conf_h, last_br_h, rng = runner_state
                rng, conf_rng, br_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action
                act_0, (_, val_0), pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_dones["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
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
                    done=last_dones["agent_1"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
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
                new_runner_state = (train_state_conf, train_state_br, env_state_next, obs_next, done, new_conf_h, new_br_h, rng)
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
                    minbatch_ego, minbatch_br = batch_infos
                    traj_batch_ego, advantages_ego, returns_ego = minbatch_ego
                    traj_batch_br, advantages_br, returns_br = minbatch_br

                    init_conf_hstate = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                    def _loss_fn_conf(params, traj_batch_ego, gae_ego, target_v_ego, traj_batch_br, gae_br, target_v_br):
                        # get policy and value of confederate versus ego and best response agents respectively

                        _, (value_ego, _), pi_ego, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_ego.obs, 
                            done=traj_batch_ego.done,
                            avail_actions=traj_batch_ego.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )
                        _, (_, value_br), pi_br, _ = confederate_policy.get_action_value_policy(
                            params=params, 
                            obs=traj_batch_br.obs, 
                            done=traj_batch_br.done,
                            avail_actions=traj_batch_br.avail_actions,
                            hstate=init_conf_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is not used here 
                        )

                        log_prob_ego = pi_ego.log_prob(traj_batch_ego.action)
                        log_prob_br = pi_br.log_prob(traj_batch_br.action)

                        # Value loss for interaction with ego agent
                        value_pred_ego_clipped = traj_batch_ego.value + (
                            value_ego - traj_batch_ego.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_ego = jnp.square(value_ego - target_v_ego)
                        value_losses_clipped_ego = jnp.square(value_pred_ego_clipped - target_v_ego)
                        value_loss_ego = (
                            0.5 * jnp.maximum(value_losses_ego, value_losses_clipped_ego).mean()
                        )

                        # Value loss for interaction with best response agent
                        value_pred_br_clipped = traj_batch_br.value + (
                            value_br - traj_batch_br.value
                            ).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_br = jnp.square(value_br - target_v_br)
                        value_losses_clipped_br = jnp.square(value_pred_br_clipped - target_v_br)
                        value_loss_br = (
                            0.5 * jnp.maximum(value_losses_br, value_losses_clipped_br).mean()
                        )

                        # Policy gradient loss for interaction with ego agent
                        ratio_ego = jnp.exp(log_prob_ego - traj_batch_ego.log_prob)
                        gae_norm_ego = (gae_ego - gae_ego.mean()) / (gae_ego.std() + 1e-8)
                        pg_loss_1_ego = ratio_ego * gae_norm_ego
                        pg_loss_2_ego = jnp.clip(
                            ratio_ego, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm_ego
                        pg_loss_ego = -jnp.mean(jnp.minimum(pg_loss_1_ego, pg_loss_2_ego))

                        # Policy gradient loss for interaction with best response agent
                        ratio_br = jnp.exp(log_prob_br - traj_batch_br.log_prob)
                        gae_norm_br = (gae_br - gae_br.mean()) / (gae_br.std() + 1e-8)
                        pg_loss_1_br = ratio_br * gae_norm_br
                        pg_loss_2_br = jnp.clip(
                            ratio_br, 
                            1.0 - config["CLIP_EPS"], 
                            1.0 + config["CLIP_EPS"]) * gae_norm_br
                        pg_loss_br = -jnp.mean(jnp.minimum(pg_loss_1_br, pg_loss_2_br))

                        # Entropy for interaction with ego agent
                        entropy_ego = jnp.mean(pi_ego.entropy())
                        
                        # Entropy for interaction with best response agent
                        entropy_br = jnp.mean(pi_br.entropy())

                        total_loss = (pg_loss_ego + pg_loss_br) + config["VF_COEF"] * (value_loss_ego + value_loss_br) - config["ENT_COEF"] * (entropy_ego+entropy_br)
                        return total_loss, (value_loss_ego, value_loss_br, pg_loss_ego, pg_loss_br, entropy_ego, entropy_br)

                    grad_fn = jax.value_and_grad(_loss_fn_conf, has_aux=True)
                    (loss_val, aux_vals), grads = grad_fn(
                        train_state_conf.params, 
                        traj_batch_ego, advantages_ego, returns_ego, 
                        traj_batch_br, advantages_br, returns_br)
                    train_state_conf = train_state_conf.apply_gradients(grads=grads)
                    return train_state_conf, (loss_val, aux_vals)
                
                def _update_minbatch_br(train_state_br, batch_info):
                    traj_batch_br, advantages, returns = batch_info
                    init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
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
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss
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


                (
                    train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br, 
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br
                ) = update_state

                rng_ego, perm_rng_ego, perm_rng_br0, perm_rng_br1 = jax.random.split(rng_ego, 4)

                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size_ego = config["MINIBATCH_SIZE_EGO"] * config["NUM_MINIBATCHES"] // 2 
                batch_size_br = config["MINIBATCH_SIZE_BR"] * config["NUM_MINIBATCHES"] // 2 
                assert (
                    batch_size_ego == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"

                permutation_ego = jax.random.permutation(perm_rng_ego, batch_size_ego)
                permutation_br0 = jax.random.permutation(perm_rng_br0, batch_size_br)
                permutation_br1 = jax.random.permutation(perm_rng_br1, batch_size_br)

                batch_ego = (traj_batch_ego, advantages_conf_ego, targets_conf_ego)
                batch_br0 = (traj_batch_br_0, advantages_conf_br, targets_conf_br)
                batch_br1 = (traj_batch_br_1, advantages_br, targets_br)
                
                batch_ego_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_ego,) + x.shape[2:]), batch_ego
                )
                batch_br0_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_br0
                )
                batch_br1_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_br1
                )

                shuffled_batch_ego = jax.tree.map(
                    lambda x: jnp.take(x, permutation_ego, axis=0), batch_ego_reshaped
                )
                shuffled_batch_br0 = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br0, axis=0), batch_br0_reshaped
                )
                shuffled_batch_br1 = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br1, axis=0), batch_br1_reshaped
                )

                minibatches_ego = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_ego,
                )

                minibatches_br0 = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_br0,
                )

                minibatches_br1 = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_br1,
                )

                # Update confederate
                train_state_conf, total_loss = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_ego, minibatches_br0)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, minibatches_br1
                )

                update_state = (train_state_conf, train_state_br, 
                    traj_batch_ego, traj_batch_br_0, traj_batch_br_1, 
                    advantages_conf_ego, advantages_conf_br, advantages_br,
                    targets_conf_ego, targets_conf_br, targets_br, 
                    rng_ego, rng_br)
                return update_state, (total_loss, total_loss_br)

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
                    rng_ego, rng_br, update_steps
                ) = update_runner_state

                # 1) rollout for interactions against ego agent
                runner_state_ego = (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                                    conf_hstate_ego, ego_hstate, rng_ego)
                runner_state_ego, traj_batch_ego = jax.lax.scan(
                    _env_step_ego, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_ego, last_obs_ego, last_dones_ego, 
                 conf_hstate_ego, ego_hstate, rng_ego) = runner_state_ego

                # 2) rollout for interactions against br agent
                runner_state_br = (train_state_conf, train_state_br, env_state_br, last_obs_br, 
                                   last_dones_br, conf_hstate_br, br_hstate, rng_br)
                runner_state_br, traj_batch_br = jax.lax.scan(
                    _env_step_br, runner_state_br, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_br, last_obs_br, last_dones_br, 
                conf_hstate_br, br_hstate, rng_br) = runner_state_br

                traj_batch_br_0, traj_batch_br_1 = traj_batch_br

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
                    rng_ego, rng_br
                )
                update_state, all_losses = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state_conf = update_state[0]
                train_state_br = update_state[1]

                # Metrics
                metric = traj_batch_ego.info
                metric["update_steps"] = update_steps
                metric["value_loss_conf_against_ego"] = all_losses[0][1][0]
                metric["value_loss_conf_against_br"] = all_losses[0][1][1]
                metric["pg_loss_conf_against_ego"] = all_losses[0][1][2]
                metric["pg_loss_conf_against_br"] = all_losses[0][1][3]
                metric["entropy_conf_against_ego"] = all_losses[0][1][4]
                metric["entropy_conf_against_br"] = all_losses[0][1][5]
                metric["average_rewards_br"] = jnp.mean(traj_batch_br_0.reward)
                metric["average_rewards_ego"] = jnp.mean(traj_batch_ego.reward)

                #value_loss_br, pg_loss_br, entropy_loss_br = total_loss_br[1]
                metric["value_loss_br"] = all_losses[1][1][0]
                metric["pg_loss_br"] = all_losses[1][1][1]
                metric["entropy_loss_br"] = all_losses[1][1][2]

                new_runner_state = (
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate,  
                    rng_ego, rng_br, update_steps + 1
                )
                return (new_runner_state, metric)

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
                ((
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br , last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps),
                    None
                )

                (
                    train_state_conf, train_state_br, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_ego, last_dones_br, 
                    conf_hstate_ego, ego_hstate, conf_hstate_br, br_hstate, 
                    rng_ego, rng_br, update_steps
                ) = new_runner_state

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
                    last_ep_info_with_ego = run_episodes(rng, env, 
                        agent_0_param=train_state_conf.params, agent_0_policy=confederate_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["NUM_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(rng, env, 
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
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
                         rng_ego, rng_br, update_steps),
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
                rng_ego, rng_br, update_steps
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
    out = train_fn(rngs)
    return out

def open_ended_training_step(carry, ego_policy, partner_population, config, env):
    '''
    Train the ego agent against the regret-maximizing partners. 
    Note: Currently training fcp agent against **all** adversarial partner checkpoints
    TODO: Limit training against the last adversarial checkpoints instead.
    '''
    prev_ego_params, rng = carry
    rng, partner_rng, ego_rng = jax.random.split(rng, 3)
    
    # Train partner agents with ego_policy
    train_out = train_regret_maximizing_partners(config, prev_ego_params, ego_policy, env, partner_rng)
    train_partner_params = train_out["checkpoints_conf"]
    
    # Reshape partner parameters for AgentPopulation
    pop_size = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]

    
    # Flatten partner parameters for AgentPopulation
    flattened_partner_params = jax.tree.map(
        lambda x: x.reshape((pop_size,) + x.shape[2:]), 
        train_partner_params
    )
    
    # Train ego agent using train_ppo_ego_agent
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
    # remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, rng)
    return carry, (train_out, ego_out)


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

    num_open_ended_iters = ego_metrics["returned_episode_returns"].shape[0]
    num_partner_updates = teammate_metrics["returned_episode_returns"].shape[2]
    num_ego_updates = ego_metrics["returned_episode_returns"].shape[2]

    # Process/extract PAIRED-specific losses    
    # shape (num_open_ended_iters, num_partner_seeds, num_updates, num_eval_episodes, num_agents_per_env)
    avg_teammate_sp_returns = np.asarray(teammate_metrics["eval_ep_last_info_br"]["returned_episode_returns"])[..., 0].mean(axis=(1, 3))
    avg_teammate_xp_returns = np.asarray(teammate_metrics["eval_ep_last_info_ego"]["returned_episode_returns"])[..., 0].mean(axis=(1, 3))

    # Conf vs ego, conf vs br, br losses
    #  shape (num_open_ended_iters, num_partner_seeds, num_updates, update_epochs, num_minibatches)
    avg_value_losses_teammate_against_ego = np.asarray(teammate_metrics["value_loss_conf_against_ego"]).mean(axis=(1, 3, 4))
    avg_value_losses_teammate_against_br = np.asarray(teammate_metrics["value_loss_conf_against_br"]).mean(axis=(1, 3, 4)) 
    avg_value_losses_br = np.asarray(teammate_metrics["value_loss_br"]).mean(axis=(1, 3, 4))
    
    avg_actor_losses_teammate_against_ego = np.asarray(teammate_metrics["pg_loss_conf_against_ego"]).mean(axis=(1, 3, 4)) 
    avg_actor_losses_teammate_against_br = np.asarray(teammate_metrics["pg_loss_conf_against_br"]).mean(axis=(1, 3, 4))
    avg_actor_losses_br = np.asarray(teammate_metrics["pg_loss_br"]).mean(axis=(1, 3, 4))
    
    avg_entropy_losses_teammate_against_ego = np.asarray(teammate_metrics["entropy_conf_against_ego"]).mean(axis=(1, 3, 4))
    avg_entropy_losses_teammate_against_br = np.asarray(teammate_metrics["entropy_conf_against_br"]).mean(axis=(1, 3, 4))
    avg_entropy_losses_br = np.asarray(teammate_metrics["entropy_loss_br"]).mean(axis=(1, 3, 4))
    
    # shape (num_open_ended_iters, num_partner_seeds, num_updates)
    avg_rewards_teammate_against_br = np.asarray(teammate_metrics["average_rewards_br"]).mean(axis=1)
    avg_rewards_teammate_against_ego = np.asarray(teammate_metrics["average_rewards_ego"]).mean(axis=1)
    
    # Process ego-specific metrics
    # shape (num_open_ended_iters, num_ego_seeds, num_updates, num_partners, num_eval_episodes, num_agents_per_env)
    avg_ego_returns = np.asarray(ego_metrics["eval_ep_last_info"]["returned_episode_returns"])[..., 0].mean(axis=(1, 3, 4))
    # shape (num_open_ended_iters, num_ego_seeds, num_updates, update_epochs, num_minibatches)
    avg_ego_value_losses = np.asarray(ego_metrics["value_loss"]).mean(axis=(1, 3, 4))
    avg_ego_actor_losses = np.asarray(ego_metrics["actor_loss"]).mean(axis=(1, 3, 4))
    avg_ego_entropy_losses = np.asarray(ego_metrics["entropy_loss"]).mean(axis=(1, 3, 4))
    
    for iter_idx in range(num_open_ended_iters):
        ### Teammate metrics processing
        # Extract teammate-vs-ego metrics
        teammate_metrics_iter = jax.tree.map(lambda x: x[iter_idx], teammate_metrics)
        teammate_stats = get_stats(teammate_metrics_iter, metric_names)
        teammate_stats = {k: np.mean(np.array(v), axis=0) for k, v in teammate_stats.items()}
        
        # Log all partner metrics
        for step in range(num_partner_updates):
            global_step = iter_idx * num_partner_updates + step
            
            # Log standard partner stats from get_stats
            for stat_name, stat_data in teammate_stats.items():
                if step < stat_data.shape[0]:  # Ensure step is within bounds
                    stat_mean = stat_data[step, 0]
                    logger.log_item(f"Train/Conf-Against-Ego_{stat_name}", stat_mean, train_step=global_step)
            
            # Log paired-specific metrics
            # Eval metrics
            logger.log_item("Eval/ConfReturn-Against-Ego", avg_teammate_xp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/ConfReturn-Against-BR", avg_teammate_sp_returns[iter_idx][step], train_step=global_step)
            logger.log_item("Eval/EgoReturn-Against-Conf", avg_ego_returns[iter_idx][step], train_step=global_step)
            
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
        
            # Rewards
            logger.log_item("Losses/AvgConfEgoRewards", avg_rewards_teammate_against_ego[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/AvgConfBRRewards", avg_rewards_teammate_against_br[iter_idx][step], train_step=global_step)

        ### Ego metrics processing
        # Extract ego train stats
        ego_metrics_iter = jax.tree.map(lambda x: x[iter_idx], ego_metrics)
        ego_stats = get_stats(ego_metrics_iter, metric_names)
        ego_stats = {k: np.mean(np.array(v), axis=0) for k, v in ego_stats.items()}
        
        for step in range(num_ego_updates):
            global_step = iter_idx * num_ego_updates + step
            # Standard ego stats from get_stats
            for stat_name, stat_data in ego_stats.items():
                if step < stat_data.shape[0]:  # Ensure step is within bounds
                    stat_mean = stat_data[step, 0]
                    logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=global_step)

            # Ego agent losses
            logger.log_item("Losses/EgoValueLoss", avg_ego_value_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoActorLoss", avg_ego_actor_losses[iter_idx][step], train_step=global_step)
            logger.log_item("Losses/EgoEntropyLoss", avg_ego_entropy_losses[iter_idx][step], train_step=global_step)
            
    logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        os.remove(out_savepath)
        
def run_paired(config):
    algorithm_config = dict(config["algorithm"])
    wandb_logger = Logger(config)

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    # Initialize ego agent using initialize_s5_agent from ppo_ego.py
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    
    # Initialize partner policy once - reused for all iterations
    partner_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0]
    )
    
    # Create partner population
    partner_population = AgentPopulation(
        pop_size=algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"],
        policy_cls=partner_policy
    )

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, partner_population, algorithm_config, env)
    
    init_carry = (init_ego_params, train_rng)
    
    log.info("Starting open-ended PAIRED training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        final_carry, outs = jax.lax.scan(
            open_ended_step_fn, 
            init_carry, 
            xs=None,
            length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
        )
    
    end_time = time.time()
    log.info(f"Open-ended PAIRED training completed in {end_time - start_time} seconds.")

    # Run heldout evaluation 
    log.info("Running heldout evaluation...")
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, 0, -1], ego_outs["checkpoints"]) # shape (num_open_ended_iters, 1, num_ckpts, leaf_dim)
    heldout_eval_metrics, ego_names, heldout_names = run_heldout_evaluation(config, ego_policy, ego_params, init_ego_params)

    # Log metrics
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)
    log_heldout_metrics(config, wandb_logger, heldout_eval_metrics, ego_names, heldout_names, metric_names)

    # Cleanup
    wandb_logger.close()

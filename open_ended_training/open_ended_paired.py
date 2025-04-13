import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from envs.log_wrapper import LogWrapper


from envs import make_env
from ppo.ippo import unbatchify, Transition
from common.mlp_actor_critic import ActorCritic, ActorWithDoubleCritic
from common.s5_actor_critic import S5ActorCritic, StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from common.wandb_visualizations import Logger
from common.agent_interface import AgentPopulation, ActorWithDoubleCriticPolicy, S5ActorCriticPolicy, MLPActorCriticPolicy
from ego_agent_training.ppo_ego import train_ppo_ego_agent, initialize_s5_agent
from ego_agent_training.run_episodes import run_episodes

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

            # # S5 specific parameters
            # d_model = config["S5_D_MODEL"]
            # ssm_size = config["S5_SSM_SIZE"]
            # n_layers = config["S5_N_LAYERS"]
            # blocks = config["S5_BLOCKS"]
            # block_size = int(ssm_size / blocks)

            # Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
            # block_size = block_size // 2
            # ssm_size = ssm_size // 2
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            # Vinv = V.conj().T

            # ssm_init_fn = init_S5SSM(H=d_model,
            #                         P=ssm_size,
            #                         Lambda_re_init=Lambda.real,
            #                         Lambda_im_init=Lambda.imag,
            #                         V=V,
            #                         Vinv=Vinv)
            
            # ego_policy = S5ActorCritic(env.action_space(env.agents[0]).n, 
            #                            ssm_init_fn=ssm_init_fn,
            #                            fc_hidden_dim=config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
            #                            ssm_hidden_dim=config["S5_SSM_SIZE"],)

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
                # TODO: pass BOTH dones through the step and give them to the appropriate policies
                train_state_conf, env_state, last_obs, last_dones, last_ego_h, last_conf_h, rng = runner_state
                rng, act_rng, partner_rng, step_rng = jax.random.split(rng, 4)

                obs_0 = last_obs["agent_0"]
                obs_1 = last_obs["agent_1"]

                # Get available actions for agent 0 from environment state
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                # Agent_0 (confederate) action using policy interface
                last_conf_dones = last_dones["agent_0"]
                act_0, val_0, pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_conf_dones.reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # confederate has same done status
                    avail_actions=avail_actions_0,
                    hstate=last_conf_h,
                    rng=act_rng
                )
                act_0 = act_0.squeeze()
                logp_0 = pi_0.log_prob(act_0)

                # Agent_1 (ego) action using policy interface
                last_ego_dones = last_dones["agent_1"]
                act_1, _, _, new_ego_h = ego_policy.get_action_value_policy(
                    params=ego_params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_ego_dones.reshape(1, config["NUM_CONTROLLED_ACTORS"]),
                    avail_actions=avail_actions_1,
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
                new_runner_state = (train_state_conf, env_state_next, obs_next, done, new_ego_h, new_conf_h, rng)
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
                avail_actions = jax.lax.stop_gradient(avail_actions)
                avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
                avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

                last_conf_dones = last_dones["agent_0"]
                # Agent_0 (confederate) action
                act_0, val_0, pi_0, new_conf_h = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=obs_0.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_conf_dones.reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
                    avail_actions=avail_actions_0,
                    hstate=last_conf_h,
                    rng=conf_rng
                )
                act_0 = act_0.squeeze()
                logp_0 = pi_0.log_prob(act_0)


                # Agent 1 (best response) action
                last_br_dones = last_dones["agent_1"]
                act_1, val_1, pi_1, new_br_h = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=obs_1.reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
                    done=last_br_dones.reshape(1, config["NUM_CONTROLLED_ACTORS"]),  # Get done flag with right shape
                    avail_actions=avail_actions_1,
                    hstate=last_br_h,
                    rng=br_rng
                )
                act_1 = act_1.squeeze()
                logp_1 = pi_1.log_prob(act_1)

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

                    def _loss_fn_conf(params, traj_batch_ego, gae_ego, target_v_ego, traj_batch_br, gae_br, target_v_br):
                        # get policy and value of confederate versus ego and best response agents respectively
                        # TODO: REPLACE THIS WITH THE GET_ACTION_VALUE_POLICY FUNCTION
                        pi_ego, value_ego, _ = confederate_policy.network.apply(params, (traj_batch_ego.obs, traj_batch_ego.avail_actions))
                        pi_br, _, value_br = confederate_policy.network.apply(params, (traj_batch_br.obs, traj_batch_br.avail_actions))
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
                        # total_loss = pg_loss_ego + config["VF_COEF"] * value_loss_ego - config["ENT_COEF"] * entropy_ego

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

                    def _loss_fn_br(params, traj_batch_br, gae, target_v):
                        # TODO: REPLACE THIS WITH THE GET_ACTION_VALUE_POLICY FUNCTION
                        pi, value = br_policy.network.apply(params, (traj_batch_br.obs, traj_batch_br.avail_actions))
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

                rng_ego, perm_rng_conf = jax.random.split(rng_ego)
                rng_ego, perm_rng_conf2 = jax.random.split(rng_ego)
                rng_br, perm_rng_br = jax.random.split(rng_br)

                # Divide batch size by TWO because we are only training on data of agent_0
                batch_size_ego = config["MINIBATCH_SIZE_EGO"] * config["NUM_MINIBATCHES"] // 2 
                batch_size_br = config["MINIBATCH_SIZE_BR"] * config["NUM_MINIBATCHES"] // 2 
                assert (
                    batch_size_ego == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"
                assert (
                    batch_size_br == config["ROLLOUT_LENGTH"] * config["NUM_ACTORS"] // 2
                ), "batch size must be equal to number of steps * number of actors"

                permutation_conf1 = jax.random.permutation(perm_rng_conf, batch_size_ego)
                permutation_conf2 = jax.random.permutation(perm_rng_conf2, batch_size_br)
                permutation_br = jax.random.permutation(perm_rng_br, batch_size_br)

                batch_ego = (traj_batch_ego, advantages_conf_ego, targets_conf_ego)
                batch_agent_0_br = (traj_batch_br_0, advantages_conf_br, targets_conf_br)
                batch_agent_1_br = (traj_batch_br_1, advantages_br, targets_br)
                
                batch_ego_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_ego,) + x.shape[2:]), batch_ego
                )
                batch_agent_0_br_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_agent_0_br
                )
                batch_agent_1_br_reshaped = jax.tree.map(
                    lambda x: x.reshape((batch_size_br,) + x.shape[2:]), batch_agent_1_br
                )

                shuffled_batch_ego = jax.tree.map(
                    lambda x: jnp.take(x, permutation_conf1, axis=0), batch_ego_reshaped
                )
                shuffled_batch_0_br = jax.tree.map(
                    lambda x: jnp.take(x, permutation_conf2, axis=0), batch_agent_0_br_reshaped
                )
                shuffled_batch_1_br = jax.tree.map(
                    lambda x: jnp.take(x, permutation_br, axis=0), batch_agent_1_br_reshaped
                )

                minibatches_ego = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_ego,
                )

                minibatches_0_br = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_0_br,
                )

                minibatches_1_br = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch_1_br,
                )

                # Update confederate
                train_state_conf, total_loss = jax.lax.scan(
                    _update_minbatch_conf, train_state_conf, (minibatches_ego, minibatches_0_br)
                )

                # Update best response
                train_state_br, total_loss_br = jax.lax.scan(
                    _update_minbatch_br, train_state_br, minibatches_1_br
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
                # TODO: make are passing thru dones correctly 
                DONE_PLACEHOLDER = None
                (
                    train_state_conf, train_state_br, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
                    br_hstate, rng_ego, rng_br, update_steps
                ) = update_runner_state

                # 1) rollout for interactions against ego agent
                runner_state_ego = (train_state_conf, env_state_ego, last_obs_ego, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, rng_ego)
                runner_state_ego, traj_batch_ego = jax.lax.scan(
                    _env_step_ego, runner_state_ego, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, env_state_ego, last_obs_ego, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, rng_ego) = runner_state_ego

                # 2) rollout for interactions against br agent
                runner_state_br = (train_state_conf, train_state_br, env_state_br, last_obs_br, 
                                   last_dones_partner, DONE_PLACEHOLDER, conf_hstate_br, br_hstate, rng_br)
                runner_state_br, traj_batch_br = jax.lax.scan(
                    _env_step_br, runner_state_br, None, config["ROLLOUT_LENGTH"])
                (train_state_conf, train_state_br, env_state_br, last_obs_br, DONE_PLACEHOLDER, conf_hstate_br, br_hstate, rng_br) = runner_state_br

                # Get agent 0 and agent 1 trajectories from interaction between conf policy and its BR policy.
                traj_batch_br_0, traj_batch_br_1 = traj_batch_br

                # 3a) compute advantage for confederate agent from interaction with ego agent
                last_obs_batch_0_ego = last_obs_ego["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state_ego.env_state)["agent_0"].astype(jnp.float32)
                
                # Get last value using agent interface
                _, last_val_0_ego, _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_batch_0_ego, # TODO: check if we need to reshape this
                    done=DONE_PLACEHOLDER, # TODO: add this
                    avail_actions=avail_actions_0,
                    hstate=conf_hstate_ego,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                advantages_conf_ego, targets_conf_ego = _calculate_gae(traj_batch_ego, last_val_0_ego)

                # 3b) compute advantage for confederate agent from interaction with br policy
                last_obs_batch_0_br = last_obs_br["agent_0"]
                # Get available actions for agent 0 from environment state
                avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_0"].astype(jnp.float32)
                
                # Get last value using agent interface
                _, last_val_0_br, _, _ = confederate_policy.get_action_value_policy(
                    params=train_state_conf.params,
                    obs=last_obs_batch_0_br, # TODO: check if we need to reshape this
                    done=DONE_PLACEHOLDER, # TODO: add this
                    avail_actions=avail_actions_0,
                    hstate=conf_hstate_br,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                advantages_conf_br, targets_conf_br = _calculate_gae(traj_batch_br_0, last_val_0_br)

                # 3c) compute advantage for br policy from interaction with confederate agent
                last_obs_batch_1_br = last_obs_br["agent_1"]
                avail_actions_1 = jax.vmap(env.get_avail_actions)(env_state_br.env_state)["agent_1"].astype(jnp.float32)
                # Get last value using agent interface
                _, last_val_1_br, _, _ = br_policy.get_action_value_policy(
                    params=train_state_br.params,
                    obs=last_obs_batch_1_br, # TODO: check if we need to reshape this
                    done=DONE_PLACEHOLDER, # TODO: add this
                    avail_actions=avail_actions_1,
                    hstate=br_hstate,
                    rng=jax.random.PRNGKey(0)  # dummy key as we don't sample actions
                )
                advantages_br, targets_br = _calculate_gae(traj_batch_br_1, last_val_1_br)

                # 3) PPO update
                update_state = (
                    train_state_conf, train_state_br, traj_batch_ego, 
                    traj_batch_br_0, traj_batch_br_1, 
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

                # Reset environment to prepare for the next update step
                # TODO: make sure the chunk of code until #DONE is correct
                rng_ego, reset_rng_ego = jax.random.split(rng_ego)
                rng_br, reset_rng_br = jax.random.split(rng_br)
                reset_rngs_ego = jax.random.split(reset_rng_ego, config["NUM_ENVS"])
                reset_rngs_br = jax.random.split(reset_rng_br, config["NUM_ENVS"])
                
                obsv_ego, env_state_ego = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_ego)
                obsv_br, env_state_br = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_br)
                
                # Reset hidden states
                conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                conf_hstate_br = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
                
                # Reset done flags
                last_dones_partner = jnp.zeros_like(last_dones_partner)
                # DONE

                new_runner_state = (
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    obsv_ego, obsv_br, last_dones_partner, 
                    DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
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
            
            # max_episode_steps_br = config["ROLLOUT_LENGTH"]
            # max_episode_steps_ego = config["ROLLOUT_LENGTH"]

            # def run_single_episode_br(ep_rng, br_param, conf_param):
            #     '''agent_0 is the confederate, agent 1 is the best response'''
            #     # Reset the env.
            #     ep_rng, reset_rng = jax.random.split(ep_rng)
            #     obs, env_state = env.reset(reset_rng)
            #     # Get available actions for agent 0 from environment state
            #     avail_actions = env.get_avail_actions(env_state.env_state)
            #     avail_actions = jax.lax.stop_gradient(avail_actions)
            #     avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            #     avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
            #     against_br_return = jnp.zeros(1, dtype=float)
                
            #     # Do one step to get a dummy info structure.
            #     ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
            #     # TODO use the get_action function to get the action here
            #     pi0, _, _ = confederate_policy.network.apply(conf_param, (obs["agent_0"], avail_actions_0))
            #     # for LBF, IPPO policies do better when sampled than when taking the mode. 
            #     act0 = pi0.sample(seed=act_rng)
            #     br_agent_net = ActorCritic(env.action_space(env.agents[1]).n)
            #     pi1, _ = br_agent_net.apply(br_param, (obs["agent_1"], avail_actions_1))
            #     act1 = pi1.sample(seed=part_rng)
                    
            #     both_actions = [act0, act1]
            #     env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            #     _, _, reward, done, dummy_info = env.step(step_rng, env_state, env_act)
            #     against_br_return = against_br_return + reward["agent_0"]

            #     # We'll use a scan to iterate steps until the episode is done.
            #     ep_ts = 1
            #     ep_rng, remaining_steps_rng = jax.random.split(ep_rng)
            #     init_carry = (ep_ts, env_state, obs, remaining_steps_rng, done, dummy_info, against_br_return)
            #     def scan_step(carry, _):
            #         def take_step(carry_step):
            #             ep_ts, env_state, obs, ep_rng, done, last_info, against_br_return = carry_step
            #             ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
                        
            #             # Get available actions for agent 0 from environment state
            #             avail_actions = env.get_avail_actions(env_state.env_state)
            #             avail_actions = jax.lax.stop_gradient(avail_actions)
            #             avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            #             avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            #             pi0, _, _ = confederate_policy.network.apply(conf_param, (obs["agent_0"], avail_actions_0))
            #             act0 = pi0.sample(seed=act_rng) # sample because mode does worse on LBF

            #             pi1, _ = br_agent_net.apply(br_param, (obs["agent_1"], avail_actions_1))
            #             act1 = pi1.sample(seed=part_rng)

            #             both_actions = [act0, act1]
            #             env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}

            #             obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)
            #             against_br_return = against_br_return + reward["agent_0"]

            #             return (ep_ts + 1, env_state_next, obs_next, ep_rng, done_next, info_next, against_br_return)
                            
            #         ep_ts, env_state, obs, ep_rng, done, last_info, against_br_return = carry
            #         new_carry = jax.lax.cond(
            #             done["__all__"],
            #             # if done, execute true function(operand). else, execute false function(operand).
            #             lambda curr_carry: curr_carry, # True fn
            #             take_step, # False fn
            #             operand=carry
            #         )
            #         return new_carry, None

            #     final_carry, _ = jax.lax.scan(
            #         scan_step, init_carry, None, length=max_episode_steps_br)
            #     # Return the final info (which includes the episode return via LogWrapper).
            #     return (final_carry[-2], final_carry[-1])
            
            # def run_single_episode_ego(ep_rng, ego_param, conf_param):
            #     '''agent 0 is the confederate, agent 1 is the ego'''
            #     # Reset the env.
            #     ep_rng, reset_rng = jax.random.split(ep_rng)
            #     obs, env_state = env.reset(reset_rng)
            #     against_ego_return = jnp.zeros(1, dtype=float)

            #     # Get available actions for agent 0 from environment state
            #     avail_actions = env.get_avail_actions(env_state.env_state)
            #     avail_actions = jax.lax.stop_gradient(avail_actions)
            #     avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            #     avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            #     # Do one step to get a dummy info structure.
            #     ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
            #     pi0, _, _ = confederate_policy.network.apply(conf_param, (obs["agent_0"], avail_actions_0))
            #     # for LBF, IPPO policies do better when sampled than when taking the mode. 
            #     act0 = pi0.sample(seed=act_rng)

            #     hstate_ego = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
            #     init_done = jnp.zeros(1, dtype=bool)
            #     rnn_input_1 = (
            #         obs["agent_1"].reshape(1, 1, -1),
            #         init_done.reshape(1, 1), 
            #         avail_actions_1.reshape(1, -1)
            #     )
            #     hstate_ego, pi1, _ = ego_policy.apply(ego_param, hstate_ego, rnn_input_1)
            #     act1 = pi1.sample(seed=part_rng).squeeze()
                    
            #     both_actions = [act0, act1]
            #     env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            #     _, _, reward, done, dummy_info = env.step(step_rng, env_state, env_act)
            #     against_ego_return = against_ego_return + reward["agent_0"]

            #     # We'll use a scan to iterate steps until the episode is done.
            #     ep_ts = 1
            #     ep_rng, remaining_steps_rng = jax.random.split(ep_rng)
            #     init_carry = (ep_ts, env_state, obs, remaining_steps_rng, done, hstate_ego, dummy_info, against_ego_return)
            #     def scan_step(carry, _):
            #         def take_step(carry_step):
            #             ep_ts, env_state, obs, ep_rng, done, hstate_ego, last_info, against_ego_return = carry_step
            #             ep_rng, act_rng, part_rng, step_rng = jax.random.split(ep_rng, 4)
            #             # Get available actions for agent 0 from environment state
            #             avail_actions = env.get_avail_actions(env_state.env_state)
            #             avail_actions = jax.lax.stop_gradient(avail_actions)
            #             avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            #             avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)
                        
            #             pi0, _, _ = confederate_policy.network.apply(conf_param, (obs["agent_0"], avail_actions_0))
            #             act0 = pi0.sample(seed=act_rng) # sample because mode does worse on LBF

            #             rnn_input_1 = (
            #                 obs["agent_1"].reshape(1, 1, -1),
            #                 done["agent_1"].reshape(1, 1), 
            #                 avail_actions_1.reshape(1, -1)
            #             )
            #             hstate_ego, pi1, _ = ego_policy.apply(ego_param, hstate_ego, rnn_input_1)
            #             act1 = pi1.sample(seed=part_rng).squeeze()
            #             both_actions = [act0, act1]
            #             env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            #             obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)
            #             against_ego_return = against_ego_return + reward["agent_0"]

            #             return (ep_ts + 1, env_state_next, obs_next, ep_rng, done_next, hstate_ego, info_next, against_ego_return)
                            
            #         ep_ts, env_state, obs, ep_rng, done, hstate_ego, last_info, against_ego_return = carry
            #         new_carry = jax.lax.cond(
            #             done["__all__"],
            #             # if done, execute true function(operand). else, execute false function(operand).
            #             lambda curr_carry: curr_carry, # True fn
            #             take_step, # False fn
            #             operand=carry
            #         )
            #         return new_carry, None

            #     final_carry, _ = jax.lax.scan(
            #         scan_step, init_carry, None, length=max_episode_steps_ego)
            #     # Return the final info (which includes the episode return via LogWrapper).
            #     return (final_carry[-2], final_carry[-1]) # TODO: remove return of RETURNS

            # def run_episodes_br(ep_rng, br_param, conf_param, num_eps):
            #     def body_fn(carry, _):
            #         ep_rng = carry
            #         ep_rng, ep_rng_step = jax.random.split(ep_rng)
            #         all_outs = run_single_episode_br(ep_rng_step, br_param, conf_param)
            #         return ep_rng, all_outs
            #     ep_rng, all_outs = jax.lax.scan(body_fn, ep_rng, None, length=num_eps)
            #     return all_outs  # each leaf has shape (num_eps, ...)
            
            # def run_episodes_ego(ep_rng, ego_param, conf_param, num_eps):
            #     def body_fn(carry, _):
            #         ep_rng = carry
            #         ep_rng, ep_rng_step = jax.random.split(ep_rng)
            #         all_outs = run_single_episode_ego(ep_rng_step, ego_param, conf_param)
            #         return ep_rng, all_outs
            #     ep_rng, all_outs = jax.lax.scan(body_fn, ep_rng, None, length=num_eps)
            #     return all_outs  # each leaf has shape (num_eps, ...)

            def _update_step_with_ckpt(state_with_ckpt, unused):
                ((
                    train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br , last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
                    br_hstate, rng_ego, rng_br, update_steps
                ), checkpoint_array_conf, checkpoint_array_br, ckpt_idx, 
                    eval_info_br, eval_info_ego) = state_with_ckpt

                # Single PPO update
                (new_runner_state, metric) = _update_step(
                    (train_state_conf, train_state_br, env_state_ego, env_state_br, 
                    last_obs_ego, last_obs_br, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
                    br_hstate, rng_ego, rng_br, update_steps),
                    None
                )

                (
                    train_state_conf, train_state_br, env_state_ego, env_state_br,
                    last_obs_ego, last_obs_br, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
                    br_hstate, rng_ego, rng_br, update_steps
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
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["MAX_EVAL_EPISODES"]
                    )
                    # conf vs br
                    last_ep_info_with_br = run_episodes(rng, env, 
                        agent_0_param=train_state_br.params, agent_0_policy=br_policy,
                        agent_1_param=ego_params, agent_1_policy=ego_policy, 
                        max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["MAX_EVAL_EPISODES"]
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
                         last_obs_ego, last_obs_br, last_dones_partner, DONE_PLACEHOLDER, ego_hstate, conf_hstate_ego, conf_hstate_br, 
                         br_hstate, rng_ego, rng_br, update_steps),
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
                ego_param=train_state_conf.params, ego_policy=confederate_policy,
                partner_param=ego_params, partner_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["MAX_EVAL_EPISODES"]
            )
            ep_infos_br = run_episodes(rng_eval_br, env, 
                ego_param=train_state_br.params, ego_policy=br_policy,
                partner_param=ego_params, partner_policy=ego_policy, 
                max_episode_steps=config["ROLLOUT_LENGTH"], num_eps=config["MAX_EVAL_EPISODES"])

            # TODO: carefully examine how the hidden states are initialized
            # Initialize hidden states
            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_ego = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_conf_hstate_br = confederate_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_br_hstate = br_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])


            init_dones = jnp.zeros(config["NUM_CONTROLLED_ACTORS"], dtype=bool)
            DONE_PLACEHOLDER = jnp.zeros(config["NUM_CONTROLLED_ACTORS"], dtype=bool)
            rng, rng_ego, rng_br = jax.random.split(rng, 3)
            update_runner_state = (
                train_state_conf, train_state_br, env_state_ego, env_state_br, 
                obsv_ego, obsv_br, init_dones, DONE_PLACEHOLDER, init_ego_hstate, init_conf_hstate_ego, init_conf_hstate_br, 
                init_br_hstate, rng_ego, rng_br, update_steps
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

def open_ended_training_step(carry, ego_policy, partner_policy, config, env):
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
    
    # Create partner population
    partner_population = AgentPopulation(
        pop_params=flattened_partner_params,
        pop_size=pop_size,
        policy_cls=partner_policy
    )
    
    # Train ego agent using train_ppo_ego_agent
    ego_out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population
    )
    
    updated_ego_parameters = ego_out["final_params"]
    # remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, rng)
    return carry, (train_out, ego_out)


def process_metrics(teammate_training_logs, ego_training_logs):
    all_teammate_sp_returns = np.asarray(teammate_training_logs["metrics"]["per_iter_ep_infos_br"])
    all_teammate_xp_returns = np.asarray(teammate_training_logs["metrics"]["per_iter_ep_infos_ego"])
    all_value_losses_teammate_against_ego = np.asarray(teammate_training_logs["metrics"]["value_loss_conf_against_ego"])
    all_value_losses_teammate_against_br = np.asarray(teammate_training_logs["metrics"]["value_loss_conf_against_br"])
    all_value_losses_br = np.asarray(teammate_training_logs["metrics"]["value_loss_br"])
    all_actor_losses_teammate_against_ego = np.asarray(teammate_training_logs["metrics"]["pg_loss_conf_against_ego"])
    all_actor_losses_teammate_against_br = np.asarray(teammate_training_logs["metrics"]["pg_loss_conf_against_br"])
    all_actor_losses_br = np.asarray(teammate_training_logs["metrics"]["pg_loss_br"])
    all_entropy_losses_teammate_against_ego = np.asarray(teammate_training_logs["metrics"]["entropy_conf_against_ego"])
    all_entropy_losses_teammate_against_br = np.asarray(teammate_training_logs["metrics"]["entropy_conf_against_br"])
    all_entropy_losses_br = np.asarray(teammate_training_logs["metrics"]["entropy_loss_br"])
    br_metrics = np.asarray(teammate_training_logs["metrics"]["average_rewards_br"])
    ego_metrics = np.asarray(teammate_training_logs["metrics"]["average_rewards_ego"])

    all_ego_returns = np.asarray(ego_training_logs["metrics"]["per_iter_ep_infos"])
    all_ego_value_losses = np.asarray(ego_training_logs["metrics"]["value_loss"])
    all_ego_actor_losses = np.asarray(ego_training_logs["metrics"]["actor_loss"])
    all_ego_entropy_losses = np.asarray(ego_training_logs["metrics"]["entropy_loss"])

    average_xp_rets_per_iter = np.mean(np.mean(all_teammate_xp_returns[:, :, :, :, 0], axis=-1), axis=1)
    average_sp_rets_per_iter = np.mean(np.mean(all_teammate_sp_returns[:, :, :, :, 0], axis=-1), axis=1)
    average_ego_rets_per_iter = np.mean(np.mean(all_ego_returns[:, :, :, :, 0], axis=-1), axis=-1)

    average_value_losses_confederate_against_ego = np.mean(np.mean(np.mean(all_value_losses_teammate_against_ego, axis=-1), axis=-1), axis=1)
    average_actor_losses_confederate_against_ego = np.mean(np.mean(np.mean(all_actor_losses_teammate_against_ego, axis=-1), axis=-1), axis=1)
    average_entropy_losses_confederate_against_ego = np.mean(np.mean(np.mean(all_entropy_losses_teammate_against_ego, axis=-1), axis=-1), axis=1)
    average_value_losses_confederate_against_br = np.mean(np.mean(np.mean(all_value_losses_teammate_against_br, axis=-1), axis=-1), axis=1)
    average_actor_losses_confederate_against_br = np.mean(np.mean(np.mean(all_actor_losses_teammate_against_br, axis=-1), axis=-1), axis=1)
    average_entropy_losses_confederate_against_br = np.mean(np.mean(np.mean(all_entropy_losses_teammate_against_br, axis=-1), axis=-1), axis=1)
    average_value_losses_br = np.mean(np.mean(np.mean(all_value_losses_br, axis=-1), axis=-1), axis=1)
    average_actor_losses_br = np.mean(np.mean(np.mean(all_actor_losses_br, axis=-1), axis=-1), axis=1)
    average_entropy_losses_br = np.mean(np.mean(np.mean(all_entropy_losses_br, axis=-1), axis=-1), axis=1)
    ego_rewards = np.mean(ego_metrics, axis=1)
    br_rewards = np.mean(br_metrics, axis=1)

    average_ego_value_losses = np.mean(np.mean(all_ego_value_losses, axis=-1), axis=-1)
    average_ego_actor_losses = np.mean(np.mean(all_ego_actor_losses, axis=-1), axis=-1)
    average_ego_entropy_losses = np.mean(np.mean(all_ego_entropy_losses, axis=-1), axis=-1)

    return average_xp_rets_per_iter, average_sp_rets_per_iter, average_ego_rets_per_iter,\
          average_value_losses_confederate_against_ego, average_actor_losses_confederate_against_ego, average_entropy_losses_confederate_against_ego,\
          average_value_losses_confederate_against_br, average_actor_losses_confederate_against_br, average_entropy_losses_confederate_against_br,\
          average_ego_value_losses, average_ego_actor_losses, average_ego_entropy_losses,\
          ego_rewards, br_rewards, \
          average_value_losses_br, average_actor_losses_br, average_entropy_losses_br

def run_paired(config):
    algorithm_config = dict(config["algorithm"])
    logger = Logger(config)

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    # Initialize ego agent using initialize_s5_agent from ppo_ego.py
    ego_policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    
    # Initialize partner policy once - reused for all iterations
    partner_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0]
    )
    
    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, partner_policy, algorithm_config, env)
    
    init_carry = (init_params, train_rng)
    
    log.info("Starting open-ended PAIRED training...")
    start_time = time.time()
    with jax.disable_jit(False):
        final_carry, outs = jax.lax.scan(
            open_ended_step_fn, 
            init_carry, 
            xs=None,
            length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
        )
    
    end_time = time.time()
    log.info(f"Open-ended PAIRED training completed in {end_time - start_time} seconds.")
    
    teammate_training_logs, ego_training_logs = outs

    # postprocessed_outs = process_metrics(teammate_training_logs, ego_training_logs)

    # average_xp_rets_per_iter, average_sp_rets_per_iter, average_ego_rets_per_iter = postprocessed_outs[0], postprocessed_outs[1], postprocessed_outs[2]
    # average_value_losses_teammate_ego, average_actor_losses_teammate_ego, average_entropy_losses_teammate_ego = postprocessed_outs[3], postprocessed_outs[4], postprocessed_outs[5]
    # average_value_losses_teammate_br, average_actor_losses_teammate_br, average_entropy_losses_teammate_br = postprocessed_outs[6], postprocessed_outs[7], postprocessed_outs[8]
    # average_ego_value_losses, average_ego_actor_losses, average_ego_entropy_losses = postprocessed_outs[9], postprocessed_outs[10], postprocessed_outs[11]
    # ego_rewards, br_rewards = postprocessed_outs[12], postprocessed_outs[13],
    # average_value_losses_br, average_actor_losses_br, average_entropy_losses_br = postprocessed_outs[14], postprocessed_outs[15], postprocessed_outs[16]

    # for num_iter in range(average_xp_rets_per_iter.shape[0]):
    #     for num_step in range(average_xp_rets_per_iter.shape[1]):
    #         logger.log_item("Eval/Conf_XP", average_xp_rets_per_iter[num_iter][num_step], checkpoint=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Eval/Conf_SP", average_sp_rets_per_iter[num_iter][num_step], checkpoint=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[num_iter][num_step], checkpoint=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfValLoss-Against-Ego", average_value_losses_teammate_ego[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfActorLoss-Against-Ego", average_actor_losses_teammate_ego[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfEntropy-Against-Ego", average_entropy_losses_teammate_ego[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfValLoss-Against-BR", average_value_losses_teammate_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfActorLoss-Against-BR", average_actor_losses_teammate_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfEntropy-Against-BR",  average_entropy_losses_teammate_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/Average-BR-ValLoss", average_value_losses_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/Average-BR-ActorLoss", average_actor_losses_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/Average-BR-EntropyLoss",  average_entropy_losses_br[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageEgoValueLoss", average_ego_value_losses[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageEgoActorLoss", average_ego_actor_losses[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageEgoEntropyLoss", average_ego_entropy_losses[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfEgoRewards", ego_rewards[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)
    #         logger.log_item("Losses/AverageConfBRRewards", br_rewards[num_iter][num_step], train_step=num_iter*average_xp_rets_per_iter.shape[1] + num_step)

    #         logger.commit()
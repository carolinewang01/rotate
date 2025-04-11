'''
This file contains the code for running episodes with some ego agent and some population 
of partner agents in the environment.
'''
import jax
import jax.numpy as jnp


def run_single_episode(rng, env, ego_param, ego_policy, 
                       partner_param, partner_population, 
                       max_episode_steps
                       ):
    '''TODO: rewrite this eval code parallelize the evaluation over max_episode_steps, following
    the convention of _env_step().'''
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    init_done = jnp.zeros(1, dtype=bool)
    
    # Initialize ego hidden state
    init_hstate_0 = ego_policy.init_hstate(1)
    # Initialize partner hidden state for a single agent only
    init_partner_hstate = partner_population.init_hstate(1)

    # Get agent obses
    obs_0 = obs["agent_0"]
    obs_1 = obs["agent_1"]

    # Get available actions for agent 0 from environment state
    avail_actions = env.get_avail_actions(env_state.env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # Do one step to get a dummy info structure
    rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
    
    # Reshape inputs for S5
    obs_0_reshaped = obs_0.reshape(1, 1, -1)
    done_0_reshaped = init_done.reshape(1, 1)
    
    # Get ego action
    act_0, hstate_0 = ego_policy.get_action(
        ego_param,
        obs_0_reshaped,
        done_0_reshaped,
        avail_actions_0,
        init_hstate_0,
        act_rng
    )
    act_0 = act_0.squeeze()

    # Get partner action using the underlying policy class's get_action method directly
    act1, partner_hstate = partner_population.policy_cls.get_action(
        partner_param, 
        obs_1, 
        init_done,
        avail_actions_1,
        init_partner_hstate,  # Pass the proper hidden state
        part_rng
    )
    
    both_actions = [act_0, act1]
    env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
    _, _, _, done, dummy_info = env.step(step_rng, env_state, env_act)


    # We'll use a scan to iterate steps until the episode is done.
    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, hstate_0, partner_hstate, dummy_info)
    def scan_step(carry, _):
        def take_step(carry_step):
            ep_ts, env_state, obs, rng, done, hstate_0, partner_hstate, last_info = carry_step
            # Get available actions for agent 0 from environment state
            avail_actions = env.get_avail_actions(env_state.env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            # Get agent obses
            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            prev_done_0 = done["agent_0"]
            
            # Reshape inputs for S5
            obs_0_reshaped = obs_0.reshape(1, 1, -1)
            done_0_reshaped = prev_done_0.reshape(1, 1)
            
            # Get ego action
            rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
            act_0, hstate_0_next = ego_policy.get_action(
                ego_param,
                obs_0_reshaped,
                done_0_reshaped,
                avail_actions_0,
                hstate_0,
                act_rng
            )
            act_0 = act_0.squeeze()

            # Get partner action with proper hidden state tracking
            act1, partner_hstate_next = partner_population.policy_cls.get_action(
                partner_param, 
                obs_1, 
                prev_done_0,
                avail_actions_1,
                partner_hstate,
                part_rng
            )
            
            both_actions = [act_0, act1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, hstate_0_next, partner_hstate_next, info_next)
                
        ep_ts, env_state, obs, rng, done, hstate_0, partner_hstate, last_info = carry
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
    return final_carry[-1]

def run_episodes(rng, env, ego_param, ego_policy, partner_param, partner_population, max_episode_steps, num_eps):
    def body_fn(carry, _):
        rng = carry
        rng, ep_rng = jax.random.split(rng)
        ep_last_info = run_single_episode(ep_rng, env, ego_param, ego_policy, partner_param, partner_population, max_episode_steps)
        return rng, ep_last_info
    rng, all_outs = jax.lax.scan(body_fn, rng, None, length=num_eps)
    return all_outs  # each leaf has shape (num_eps, ...)

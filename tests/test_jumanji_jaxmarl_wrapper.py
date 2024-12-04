import os
print(os.getcwd())

import jax
import jumanji
from lbf.jumanji_jaxmarl_wrapper_v2 import JumanjiToJaxMARL
from jaxmarl.wrappers.baselines import LogWrapper

"""
The purpose of this file is to test the JumanjiToJaxMARL wrapper for the LevelBasedForaging environment.
"""

# Instantiate a Jumanji environment
env = jumanji.make('LevelBasedForaging-v0')
wrapper = JumanjiToJaxMARL(env)
wrapper = LogWrapper(wrapper)

NUM_EPISODES = 2
key = jax.random.PRNGKey(20394)

for episode in range(NUM_EPISODES):
    key, subkey = jax.random.split(key)
    obs, state = wrapper.reset(subkey)
    done = {agent: False for agent in wrapper.agents}
    done['__all__'] = False
    total_rewards = {agent: 0.0 for agent in wrapper.agents}

    while not done['__all__']:
        # Sample actions for each agent
        actions = {}
        for agent in wrapper.agents:
            action_space = wrapper.action_space(agent)
            key, action_key = jax.random.split(key)
            action = int(action_space.sample(action_key))
            actions[agent] = action

        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = wrapper.step(subkey, state, actions)


        # Process observations, rewards, dones, and info as needed
        for agent in wrapper.agents:
            total_rewards[agent] += rewards[agent]

            print(f"\nEpisode {episode}, agent {agent}, timestep {info['step_count']}")
            # Optionally print or log the rewards
            print("obs", obs[agent], "type", type(obs[agent]))
            print("rewards", rewards[agent], "type", type(rewards[agent]))
            print("dones", done[agent], "type", type(done[agent]))
            print("info", info, "type", type(info))
            # info is: 
            # info {'percent_eaten': Array(0., dtype=float32), 
            #       'returned_episode': Array([ True,  True], dtype=bool), 
            #       'returned_episode_lengths': Array([100., 100.], dtype=float32), 
            #       'returned_episode_returns': Array([0., 0.], dtype=float32), 
            #       'step_count': Array(100, dtype=int32)} type <class 'dict'>
        print("state", state, "type", type(state))

    print(f"Episode {episode} finished. Total rewards: {total_rewards}")
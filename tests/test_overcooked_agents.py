import jax
from jaxmarl.environments.overcooked import overcooked_layouts
from envs.overcooked_wrapper import OvercookedWrapper
from agents.overcooked import HeuristicAgent, StaticAgent, RandomAgent, HeuristicAgentV0
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import numpy as np
from typing import Dict, Tuple
import time

def run_episode(env, agent0, agent1, key, max_steps=400) -> Tuple[Dict[str, float], int]:
    """Run a single episode with two heuristic agents.
    
    Returns:
        Tuple containing:
        - Total rewards for each agent
        - Number of steps taken
    """
    # Reset environment
    print("Resetting environment...")
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)
    print("Environment reset complete. Observation shape:", obs["agent_0"].shape)
    
    # Initialize episode tracking
    done = {agent: False for agent in env.agents}
    done['__all__'] = False
    total_rewards = {agent: 0.0 for agent in env.agents}
    num_steps = 0
    
    # Initialize agent states
    agent0_state = agent0.initial_state
    agent1_state = agent1.initial_state
    
    state_seq = [state]
    while not done['__all__'] and num_steps < max_steps:
        # Get actions from both agents with their states
        print(f"Step {num_steps}")
        action0, agent0_state = agent0.get_action(obs["agent_0"], agent0_state)
        action1, agent1_state = agent1.get_action(obs["agent_1"], agent1_state)
        
        actions = {"agent_0": action0, "agent_1": action1}
        if action0 == 5:
            print(f"actions:", actions)
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = env.step(subkey, state, actions)
        state_seq.append(state)
        # Update rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            
        num_steps += 1
                
        # Print progress every 10 steps
        if num_steps % 10 == 0:
            print(f"Agent 0 reward: {total_rewards['agent_0']:.2f}")
            print(f"Agent 1 reward: {total_rewards['agent_1']:.2f}")
    
    return total_rewards, num_steps, state_seq

def main():
    # Initialize environment
    print("Initializing environment...")
    layout = overcooked_layouts["cramped_room"]
    env = OvercookedWrapper(
        layout=layout,
        random_reset=True,
        max_steps=100,
    )
    print("Environment initialized")
    
    # Initialize agents
    print("Initializing agents...")
    agent0 = HeuristicAgentV0("agent_0", layout=layout) # red
    agent1 = StaticAgent("agent_1") # blue
    print("Agents initialized")
    
    # Run multiple episodes
    NUM_EPISODES = 5
    VISUALIZE = True
    key = jax.random.PRNGKey(0)
    
    # Initialize returns list
    returns = []
    
    state_seq = []
    for episode in range(NUM_EPISODES):
        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
        key, subkey = jax.random.split(key)
        total_rewards, num_steps, ep_states = run_episode(env, agent0, agent1, subkey)
        state_seq += ep_states
        
        # Calculate episode return
        episode_return = sum(total_rewards.values())
        returns.append(episode_return)
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"Total steps: {num_steps}")
        print(f"Episode return: {episode_return:.2f}")
        print("Final rewards:")
        for agent in env.agents:
            print(f"  {agent}: {total_rewards[agent]:.2f}")
    
    # Print statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\nStatistics across {NUM_EPISODES} episodes:")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")

    # Visualize state sequences
    if VISUALIZE:
        viz = OvercookedVisualizer()
        for state in state_seq:
            viz.render(env.agent_view_size, state, highlight=False)
            time.sleep(.01)
        # viz.animate(state_seq, agent_view_size=5, 
        #     filename='results/overcooked/gifs/heuristic_agentv0_vs_static_agent.gif')

if __name__ == "__main__":
    main() 
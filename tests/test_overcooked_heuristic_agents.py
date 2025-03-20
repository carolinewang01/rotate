import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts
from envs.overcooked_wrapper import OvercookedWrapper
from agents.overcooked.heuristic_agents import HeuristicAgent
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import numpy as np
from typing import Dict, Tuple
import time

def run_episode(env, agent0, agent1, key, visualize: bool = False, max_steps=400) -> Tuple[Dict[str, float], int]:
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
    
    # Initialize visualizer if needed
    viz = OvercookedVisualizer() if visualize else None
    
    while not done['__all__'] and num_steps < max_steps:
        # Get actions from both agents with their states
        print(f"Step {num_steps}: Getting actions...")
        action0, agent0_state = agent0.get_action(obs["agent_0"], agent0_state)
        action1, agent1_state = agent1.get_action(obs["agent_1"], agent1_state)
        
        actions = {"agent_0": action0, "agent_1": action1}
        print(f"Step {num_steps} actions:", actions)
        
        # Step environment
        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = env.step(subkey, state, actions)
        
        # Update rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            
        num_steps += 1
        
        # Visualize state if enabled
        if visualize and num_steps % 5 == 0:  # Update visualization every 5 steps
            viz.render(env.agent_view_size, state, highlight=False)
            time.sleep(0.1)  # Small pause to allow the visualization to update
        
        # Print progress every 10 steps
        if num_steps % 10 == 0:
            print(f"\nStep {num_steps}")
            print(f"Agent 0 reward: {total_rewards['agent_0']:.2f}")
            print(f"Agent 1 reward: {total_rewards['agent_1']:.2f}")
    
    return total_rewards, num_steps

def main():
    # Initialize environment
    print("Initializing environment...")
    env = OvercookedWrapper(
        layout=overcooked_layouts["cramped_room"],
        random_reset=True,
        max_steps=400,
    )
    print("Environment initialized")
    
    # Initialize agents
    print("Initializing agents...")
    agent0 = HeuristicAgent("agent_0", map_width=env.width, map_height=env.height)
    agent1 = HeuristicAgent("agent_1", map_width=env.width, map_height=env.height)
    print("Agents initialized")
    
    # Run multiple episodes
    NUM_EPISODES = 3
    VISUALIZE = True
    key = jax.random.PRNGKey(0)
    
    # Initialize returns list
    returns = []
    
    print("Starting executing heuristic agents...")
    
    for episode in range(NUM_EPISODES):
        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
        key, subkey = jax.random.split(key)
        total_rewards, num_steps = run_episode(env, agent0, agent1, subkey, visualize=VISUALIZE)
        
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

if __name__ == "__main__":
    main() 
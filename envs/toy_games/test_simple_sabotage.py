"""
Test script for the SimpleSabotage environment.
"""

import jax
import jax.numpy as jnp
from simple_sabotage import SimpleSabotage


def test_simple_sabotage():
    """Test the SimpleSabotage environment implementation."""
    
    # Initialize environment
    env = SimpleSabotage(max_steps=5, max_history_len=10)
    
    # Test reset
    key = jax.random.PRNGKey(0)
    observations, state = env.reset(key)
    
    print("=== Testing SimpleSabotage Environment ===")
    print(f"Number of agents: {env.num_agents}")
    print(f"Agents: {env.agents}")
    print(f"Max steps: {env.max_steps}")
    print()
    
    # Check initial state
    print("Initial state:")
    print(f"  Step: {state.step}")
    print(f"  Done: {state.done}")
    print(f"  Action history shape: {state.action_history.shape}")
    print(f"  Observation shapes: {[obs.shape for obs in observations.values()]}")
    print()
    
    # Test action spaces
    print("Action spaces:")
    for agent in env.agents:
        action_space = env.action_space(agent)
        print(f"  {agent}: {action_space}")
    print()
    
    # Test observation spaces  
    print("Observation spaces:")
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        print(f"  {agent}: {obs_space}")
    print()
    
    # Test available actions
    avail_actions = env.get_avail_actions(state)
    print("Available actions:")
    for agent, actions in avail_actions.items():
        print(f"  {agent}: {actions}")
    print()
    
    # Test payoff matrix
    print("Payoff matrix:")
    action_names = ["H", "T", "S"]
    print("    ", "  ".join(action_names))
    for i, row in enumerate(env.payoff_matrix):
        print(f"{action_names[i]}:  {row}")
    print()
    
    # Run a few steps
    print("Running episode...")
    actions_sequence = [
        {"agent_0": 0, "agent_1": 0},  # H, H -> reward 1
        {"agent_0": 0, "agent_1": 1},  # H, T -> reward 0  
        {"agent_0": 1, "agent_1": 0},  # T, H -> reward 0
        {"agent_0": 1, "agent_1": 1},  # T, T -> reward 1
        {"agent_0": 2, "agent_1": 0},  # S, H -> reward -1
    ]
    
    total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    
    for step, actions in enumerate(actions_sequence):
        print(f"\nStep {step + 1}:")
        print(f"  Actions: agent_0={action_names[actions['agent_0']]}, agent_1={action_names[actions['agent_1']]}")
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones['__all__']}")
        print(f"  Current step: {state.step}")
        
        # Update total rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Check if episode is done
        if dones["__all__"]:
            print("  Episode finished!")
            break
    
    print(f"\nTotal rewards: {total_rewards}")
    
    # Test rendering
    print("\n=== Rendering ===")
    env.render(state)
    
    print("\n=== Test completed successfully! ===")


def test_sabotage_termination():
    """Test that the episode ends when either agent takes sabotage action."""
    
    print("\n=== Testing Sabotage Termination ===")
    
    env = SimpleSabotage(max_steps=10, max_history_len=10)
    key = jax.random.PRNGKey(123)
    
    observations, state = env.reset(key)
    print(f"Initial step: {state.step}")
    
    # Take a few normal actions first
    actions_sequence = [
        {"agent_0": 0, "agent_1": 1},  # H, T -> should continue
        {"agent_0": 1, "agent_1": 0},  # T, H -> should continue  
        {"agent_0": 0, "agent_1": 2},  # H, S -> should end episode!
    ]
    
    action_names = ["H", "T", "S"]
    
    for step, actions in enumerate(actions_sequence):
        print(f"\nStep {step + 1}:")
        print(f"  Actions: agent_0={action_names[actions['agent_0']]}, agent_1={action_names[actions['agent_1']]}")
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        print(f"  Rewards: {rewards}")
        print(f"  Done: {dones['__all__']}")
        print(f"  Current step: {state.step}")
        
        if dones["__all__"]:
            print(f"  Episode ended at step {step + 1}!")
            break
    
    # Test with agent_0 sabotaging
    print("\n--- Testing Agent 0 Sabotage ---")
    observations, state = env.reset(key)
    
    actions = {"agent_0": 2, "agent_1": 0}  # S, H -> should end immediately
    print(f"Actions: agent_0=S, agent_1=H")
    
    key, subkey = jax.random.split(key)
    observations, state, rewards, dones, infos = env.step(subkey, state, actions)
    
    print(f"Rewards: {rewards}")
    print(f"Done: {dones['__all__']}")
    print(f"Episode ended immediately: {dones['__all__']}")
    
    # Render to see the termination message
    env.render(state)


def test_observation_history():
    """Test that observation history is working correctly."""
    
    print("\n=== Testing Observation History ===")
    
    env = SimpleSabotage(max_steps=3, max_history_len=5)
    key = jax.random.PRNGKey(42)
    
    observations, state = env.reset(key)
    
    # Check initial observation (should be all -1s)
    initial_obs = observations["agent_0"]
    print(f"Initial observation shape: {initial_obs.shape}")
    print(f"Initial observation (first 10 elements): {initial_obs[:10]}")
    
    # Take a few actions and check history
    actions_list = [
        {"agent_0": 0, "agent_1": 1},  # H, T
        {"agent_0": 2, "agent_1": 0},  # S, H  
        {"agent_0": 1, "agent_1": 1},  # T, T
    ]
    
    for i, actions in enumerate(actions_list):
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        obs = observations["agent_0"]
        print(f"\nAfter step {i+1}:")
        print(f"  Actions taken: {actions}")
        print(f"  Action history in state: {state.action_history[:i+2]}")
        print(f"  Observation (first 10 elements): {obs[:10]}")
        
        # Decode observation back to action pairs
        obs_reshaped = obs.reshape(-1, 2)
        valid_history = obs_reshaped[obs_reshaped[:, 0] >= 0]
        print(f"  Valid history from observation: {valid_history}")


if __name__ == "__main__":
    test_simple_sabotage()
    test_sabotage_termination()
    test_observation_history()

"""
Test script for SimpleSabotage environment termination conditions.

Tests:
1. Episode ends at max_steps when no sabotage occurs
2. Episode ends immediately when either agent sabotages
"""

import jax
from simple_sabotage import SimpleSabotage


def test_no_sabotage_full_episode():
    """Test that episode runs for max_steps when no sabotage occurs."""
    print("=== Testing No Sabotage - Full Episode ===")
    
    max_steps = 5
    env = SimpleSabotage(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(0)
    
    observations, state = env.reset(key)
    print("Initial step: {}, done: {}".format(state.step, state.done))
    
    # Action sequence with no sabotage (only H and T actions)
    actions_sequence = [
        {"agent_0": 0, "agent_1": 0},  # H, H -> reward 1
        {"agent_0": 0, "agent_1": 1},  # H, T -> reward 0  
        {"agent_0": 1, "agent_1": 0},  # T, H -> reward 0
        {"agent_0": 1, "agent_1": 1},  # T, T -> reward 1
        {"agent_0": 0, "agent_1": 0},  # H, H -> reward 1
    ]
    
    action_names = ["H", "T", "S"]
    total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    
    for step, actions in enumerate(actions_sequence):
        print("\nStep {}:".format(step + 1))
        print("  Actions: agent_0={}, agent_1={}".format(action_names[actions['agent_0']], action_names[actions['agent_1']]))
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        print("  Rewards: {}".format(rewards))
        print("  Done: {}".format(dones['__all__']))
        print("  Current step: {}".format(state.step))
        
        # Update total rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Check termination conditions
        if step + 1 < max_steps:
            # Should not be done yet
            assert not dones["__all__"], "Episode ended prematurely at step {}".format(step + 1)
        else:
            # Should be done at max_steps
            assert dones["__all__"], "Episode should have ended at step {} (max_steps={})".format(step + 1, max_steps)
            assert state.step == 0, "Final step should be 0 due to JaxMARL env's auto-resetting, got {}".format(state.step)
    
    print("\nFinal total rewards: {}".format(total_rewards))
    print("Final step: {}".format(state.step))
    print("Episode completed successfully at max_steps: {}".format(state.step == max_steps))
    
    # Verify episode ended at correct step
    assert state.step == 0, f"State.step should be reset to 0, but is instead {state.step}"
    assert dones["__all__"], "Episode should be marked as done"
    
    print("> Test passed: Episode runs for exactly max_steps when no sabotage occurs")


def test_sabotage_immediate_termination():
    """Test that episode ends immediately when either agent sabotages."""
    print("\n=== Testing Sabotage - Immediate Termination ===")
    
    max_steps = 10  # Set high so we can test early termination
    env = SimpleSabotage(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(123)
    
    # Test Case 1: Agent 0 sabotages at step 3
    print("\n--- Test Case 1: Agent 0 sabotages at step 3 ---")
    observations, state = env.reset(key)
    
    actions_sequence = [
        {"agent_0": 0, "agent_1": 1},  # H, T -> should continue
        {"agent_0": 1, "agent_1": 0},  # T, H -> should continue  
        {"agent_0": 2, "agent_1": 0},  # S, H -> should end episode immediately!
    ]
    
    action_names = ["H", "T", "S"]
    total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    
    for step, actions in enumerate(actions_sequence):
        print("\nStep {}:".format(step + 1))
        print("  Actions: agent_0={}, agent_1={}".format(action_names[actions['agent_0']], action_names[actions['agent_1']]))
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        print("  Rewards: {}".format(rewards))
        print("  Done: {}".format(dones['__all__']))
        print("  Current step: {}".format(state.step))
        
        # Update total rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Check termination conditions
        if actions["agent_0"] == 2 or actions["agent_1"] == 2:
            # Episode should end immediately after sabotage
            assert dones["__all__"], "Episode should end immediately after sabotage at step {}".format(step + 1)
            assert state.step == 0, f"State.step should be reset to 0, but is instead {state.step}"
            print("  > Episode ended immediately after sabotage at step {}".format(step + 1))
            break
        else:
            # Should continue if no sabotage
            assert not dones["__all__"], "Episode ended prematurely at step {} without sabotage".format(step + 1)
    
    print("Total rewards after sabotage: {}".format(total_rewards))
    
    # Test Case 2: Agent 1 sabotages at step 1 (immediate sabotage)
    print("\n--- Test Case 2: Agent 1 sabotages at step 1 ---")
    observations, state = env.reset(key)
    
    actions = {"agent_0": 0, "agent_1": 2}  # H, S -> should end immediately
    print("Actions: agent_0=H, agent_1=S")
    
    key, subkey = jax.random.split(key)
    observations, state, rewards, dones, infos = env.step(subkey, state, actions)
    
    print("Rewards: {}".format(rewards))
    print("Done: {}".format(dones['__all__']))
    print("Current step: {}".format(state.step))
    
    # Verify immediate termination
    assert dones["__all__"], "Episode should end immediately after sabotage"
    assert state.step == 0, f"State.step should be reset to 0, but is instead {state.step}"
    assert rewards["agent_0"] == -1 and rewards["agent_1"] == -1, "Sabotage should give -1 reward, got {}".format(rewards)
    
    print("> Episode ended immediately after sabotage at step 1")
    
    # Test Case 3: Both agents sabotage simultaneously
    print("\n--- Test Case 3: Both agents sabotage simultaneously ---")
    observations, state = env.reset(key)
    
    actions = {"agent_0": 2, "agent_1": 2}  # S, S -> should end immediately
    print("Actions: agent_0=S, agent_1=S")
    
    key, subkey = jax.random.split(key)
    observations, state, rewards, dones, infos = env.step(subkey, state, actions)
    
    print("Rewards: {}".format(rewards))
    print("Done: {}".format(dones['__all__']))
    print("Current step: {}".format(state.step))
    
    # Verify immediate termination with double sabotage
    assert dones["__all__"], "Episode should end immediately when both agents sabotage"
    assert state.step == 0, f"State.step should be reset to 0, but is instead {state.step}"
    assert rewards["agent_0"] == -1 and rewards["agent_1"] == -1, "Double sabotage should give -1 reward, got {}".format(rewards)
    
    print("> Episode ended immediately when both agents sabotaged")
    print("> Test passed: Episode ends immediately upon any sabotage action")


def test_mixed_scenarios():
    """Test various mixed scenarios to ensure robustness."""
    print("\n=== Testing Mixed Scenarios ===")
    
    # Scenario 1: Sabotage at the last possible step
    print("\n--- Scenario 1: Sabotage at max_steps-1 ---")
    max_steps = 3
    env = SimpleSabotage(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(456)
    
    observations, state = env.reset(key)
    
    # Run for max_steps-1, then sabotage
    actions_sequence = [
        {"agent_0": 0, "agent_1": 0},  # H, H -> continue
        {"agent_0": 1, "agent_1": 1},  # T, T -> continue
        {"agent_0": 2, "agent_1": 0},  # S, H -> should end due to sabotage, not max_steps
    ]
    
    for step, actions in enumerate(actions_sequence):
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step(subkey, state, actions)
        
        print("Step {}: Actions={}, Done={}, Step={}".format(step + 1, actions, dones['__all__'], state.step))
        
        if step + 1 == max_steps - 1 and actions["agent_0"] != 2 and actions["agent_1"] != 2:
            # Should not be done yet (one more step to go)
            assert not dones["__all__"], "Should not be done at step {} < max_steps".format(step + 1)
        elif actions["agent_0"] == 2 or actions["agent_1"] == 2:
            # Should be done due to sabotage
            assert dones["__all__"], "Should be done due to sabotage at step {}".format(step + 1)
            print("> Episode ended due to sabotage at step {}, not max_steps".format(step + 1))
            break
        elif step + 1 == max_steps:
            # Should be done due to max_steps
            assert dones["__all__"], "Should be done due to max_steps at step {}".format(step + 1)
            break
    
    print("> Test passed: Mixed scenarios work correctly")


def run_all_tests():
    """Run all termination condition tests."""
    print("Running SimpleSabotage Termination Condition Tests")
    print("=" * 60)
    
    try:
        test_no_sabotage_full_episode()
        test_sabotage_immediate_termination()
        test_mixed_scenarios()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("> Episodes run for exactly max_steps when no sabotage occurs")
        print("> Episodes end immediately when any agent sabotages")
        print("> Termination logic works correctly in all scenarios")
        
    except AssertionError as e:
        print("\nTEST FAILED: {}".format(e))
        raise
    except Exception as e:
        print("\nUNEXPECTED ERROR: {}".format(e))
        raise


if __name__ == "__main__":
    run_all_tests()
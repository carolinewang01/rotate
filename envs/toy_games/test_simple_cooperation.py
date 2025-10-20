"""
Test script for SimpleCooperation environment termination conditions and payoff matrix.

Tests:
1. Episode ends at max_steps when no mutual cooperation occurs
2. Episode ends immediately when both agents cooperate
3. Payoff matrix works correctly for all action combinations
4. Mixed scenarios and edge cases
"""

import jax
from simple_cooperation import SimpleCooperation


def test_no_cooperation_full_episode():
    """Test that episode runs for max_steps when no mutual cooperation occurs."""
    print("=== Testing No Mutual Cooperation - Full Episode ===")
    
    max_steps = 5
    env = SimpleCooperation(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(0)
    
    observations, state = env.reset(key)
    print("Initial step: {}, done: {}".format(state.step, state.done))
    
    # Action sequence with no mutual cooperation (avoiding C, C combinations)
    actions_sequence = [
        {"agent_0": 0, "agent_1": 1},  # C, A -> reward 0
        {"agent_0": 0, "agent_1": 2},  # C, B -> reward 0  
        {"agent_0": 1, "agent_1": 0},  # A, C -> reward 0
        {"agent_0": 1, "agent_1": 1},  # A, A -> reward 0
        {"agent_0": 2, "agent_1": 2},  # B, B -> reward 0
    ]
    
    action_names = ["C", "A", "B"]
    total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    
    for step, actions in enumerate(actions_sequence):
        print("\nStep {}:".format(step + 1))
        print("  Actions: agent_0={}, agent_1={}".format(action_names[actions['agent_0']], action_names[actions['agent_1']]))
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
        
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
    
    print("\nFinal total rewards: {}".format(total_rewards))
    print("Final step: {}".format(state.step))
    
    # Verify episode ended at correct step
    assert dones["__all__"], "Episode should be marked as done"
    
    print("> Test passed: Episode runs for exactly max_steps when no mutual cooperation occurs")


def test_cooperation_immediate_termination():
    """Test that episode ends immediately when both agents cooperate."""
    print("\n=== Testing Mutual Cooperation - Immediate Termination ===")
    
    max_steps = 10  # Set high so we can test early termination
    env = SimpleCooperation(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(123)
    
    # Test Case 1: Mutual cooperation at step 3
    print("\n--- Test Case 1: Mutual cooperation at step 3 ---")
    observations, state = env.reset(key)
    
    actions_sequence = [
        {"agent_0": 0, "agent_1": 1},  # C, A -> should continue
        {"agent_0": 1, "agent_1": 0},  # A, C -> should continue  
        {"agent_0": 0, "agent_1": 0},  # C, C -> should end episode immediately!
    ]
    
    action_names = ["C", "A", "B"]
    total_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    
    for step, actions in enumerate(actions_sequence):
        print("\nStep {}:".format(step + 1))
        print("  Actions: agent_0={}, agent_1={}".format(action_names[actions['agent_0']], action_names[actions['agent_1']]))
        
        # Take step
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
        
        print("  Rewards: {}".format(rewards))
        print("  Done: {}".format(dones['__all__']))
        print("  Current step: {}".format(state.step))
        
        # Update total rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Check termination conditions
        if actions["agent_0"] == 0 and actions["agent_1"] == 0:
            # Episode should end immediately after mutual cooperation
            assert dones["__all__"], "Episode should end immediately after mutual cooperation at step {}".format(step + 1)
            print("  > Episode ended immediately after mutual cooperation at step {}".format(step + 1))
            break
        else:
            # Should continue if no mutual cooperation
            assert not dones["__all__"], "Episode ended prematurely at step {} without mutual cooperation".format(step + 1)
    
    print("Total rewards after cooperation: {}".format(total_rewards))
    
    # Test Case 2: Immediate mutual cooperation at step 1
    print("\n--- Test Case 2: Immediate mutual cooperation at step 1 ---")
    observations, state = env.reset(key)
    
    actions = {"agent_0": 0, "agent_1": 0}  # C, C -> should end immediately
    print("Actions: agent_0=C, agent_1=C")
    
    key, subkey = jax.random.split(key)
    observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
    
    print("Rewards: {}".format(rewards))
    print("Done: {}".format(dones['__all__']))
    print("Current step: {}".format(state.step))
    
    # Verify immediate termination
    assert dones["__all__"], "Episode should end immediately after mutual cooperation"
    assert rewards["agent_0"] == 1 and rewards["agent_1"] == 1, "Mutual cooperation should give reward 1, got {}".format(rewards)
    
    print("> Episode ended immediately after mutual cooperation at step 1")
    print("> Test passed: Episode ends immediately upon mutual cooperation")


def test_payoff_matrix():
    """Test that all payoff combinations work correctly according to the matrix."""
    print("\n=== Testing Payoff Matrix ===")
    
    env = SimpleCooperation(max_steps=10, max_history_len=10)
    key = jax.random.PRNGKey(456)
    
    # Expected payoff matrix: [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    expected_payoffs = {
        (0, 0): 1,  # C, C -> 1
        (0, 1): 0,  # C, A -> 0
        (0, 2): 0,  # C, B -> 0
        (1, 0): 0,  # A, C -> 0
        (1, 1): 0,  # A, A -> 0
        (1, 2): 0,  # A, B -> 0
        (2, 0): 0,  # B, C -> 0
        (2, 1): 0,  # B, A -> 0
        (2, 2): 0,  # B, B -> 0
    }
    
    action_names = ["C", "A", "B"]
    
    print("Testing all action combinations:")
    for agent0_action in range(3):
        for agent1_action in range(3):
            # Reset environment for each test
            observations, state = env.reset(key)
            
            actions = {"agent_0": agent0_action, "agent_1": agent1_action}
            expected_reward = expected_payoffs[(agent0_action, agent1_action)]
            
            key, subkey = jax.random.split(key)
            observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
            
            actual_reward = rewards["agent_0"]  # Both agents get same reward
            
            print("  {}, {} -> Expected: {}, Actual: {}, Done: {}".format(
                action_names[agent0_action], 
                action_names[agent1_action], 
                expected_reward, 
                actual_reward,
                dones['__all__']
            ))
            
            # Verify reward
            assert actual_reward == expected_reward, \
                "Wrong reward for ({}, {}): expected {}, got {}".format(
                    action_names[agent0_action], action_names[agent1_action], 
                    expected_reward, actual_reward
                )
            
            # Verify both agents get same reward
            assert rewards["agent_0"] == rewards["agent_1"], \
                "Agents should get same reward, got {}".format(rewards)
            
            # Verify termination logic
            if agent0_action == 0 and agent1_action == 0:
                assert dones["__all__"], "Episode should end when both agents cooperate (C, C)"
            # Note: We don't test non-termination here since some combinations 
            # might be at max_steps in a longer test
    
    print("> Test passed: All payoff combinations work correctly")


def test_mixed_scenarios():
    """Test various mixed scenarios to ensure robustness."""
    print("\n=== Testing Mixed Scenarios ===")
    
    # Scenario 1: Cooperation at the last possible step
    print("\n--- Scenario 1: Cooperation at max_steps-1 ---")
    max_steps = 3
    env = SimpleCooperation(max_steps=max_steps, max_history_len=10)
    key = jax.random.PRNGKey(789)
    
    observations, state = env.reset(key)
    
    # Run for max_steps-1, then cooperate
    actions_sequence = [
        {"agent_0": 1, "agent_1": 1},  # A, A -> continue
        {"agent_0": 2, "agent_1": 1},  # B, A -> continue
        {"agent_0": 0, "agent_1": 0},  # C, C -> should end due to cooperation, not max_steps
    ]
    
    for step, actions in enumerate(actions_sequence):
        key, subkey = jax.random.split(key)
        observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
        
        print("Step {}: Actions={}, Reward={}, Done={}, Step={}".format(
            step + 1, actions, rewards["agent_0"], dones['__all__'], state.step))
        
        if step + 1 < max_steps and not (actions["agent_0"] == 0 and actions["agent_1"] == 0):
            # Should not be done yet (unless cooperation happens)
            assert not dones["__all__"], "Should not be done at step {} < max_steps without cooperation".format(step + 1)
        elif actions["agent_0"] == 0 and actions["agent_1"] == 0:
            # Should be done due to cooperation
            assert dones["__all__"], "Should be done due to cooperation at step {}".format(step + 1)
            print("> Episode ended due to cooperation at step {}, not max_steps".format(step + 1))
            break
        elif step + 1 == max_steps:
            # Should be done due to max_steps
            assert dones["__all__"], "Should be done due to max_steps at step {}".format(step + 1)
            break
    
    # Scenario 2: Test observation space and history tracking
    print("\n--- Scenario 2: History tracking ---")
    env = SimpleCooperation(max_steps=5, max_history_len=3)
    key = jax.random.PRNGKey(999)
    observations, state = env.reset(key)
    
    # Check initial observation (should be all -1s for empty history)
    initial_obs = observations["agent_0"]
    expected_initial = [-1] * (3 * 2)  # max_history_len * 2 agents
    print("Initial observation shape: {}, values: {}".format(initial_obs.shape, initial_obs.tolist()))
    assert initial_obs.tolist() == expected_initial, "Initial observation should be all -1s"
    
    # Take a few steps and check history
    actions = {"agent_0": 1, "agent_1": 2}  # A, B
    key, subkey = jax.random.split(key)
    observations, state, rewards, dones, infos = env.step_env(subkey, state, actions)
    
    obs_after_step = observations["agent_0"]
    print("Observation after 1 step: {}".format(obs_after_step.tolist()))
    # First two elements should be [1, 2], rest should be -1
    expected_after_step = [1, 2, -1, -1, -1, -1]
    assert obs_after_step.tolist() == expected_after_step, \
        "Observation after step should be [1, 2, -1, -1, -1, -1], got {}".format(obs_after_step.tolist())
    
    print("> Test passed: Mixed scenarios work correctly")


def test_environment_properties():
    """Test basic environment properties and interface compliance."""
    print("\n=== Testing Environment Properties ===")
    
    env = SimpleCooperation(max_steps=5, max_history_len=10)
    
    # Test basic properties
    assert env.name == "SimpleCooperation", "Environment name should be 'SimpleCooperation'"
    assert env.num_actions == 3, "Should have 3 actions"
    assert env.num_agents == 2, "Should have 2 agents"
    assert env.agents == ["agent_0", "agent_1"], "Agent names should be ['agent_0', 'agent_1']"
    
    # Test action spaces
    for agent in env.agents:
        action_space = env.action_space(agent)
        assert action_space.n == 3, "Each agent should have 3 actions"
    
    # Test observation spaces
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        expected_obs_dim = 10 * 2  # max_history_len * 2 agents
        assert obs_space.shape == (expected_obs_dim,), \
            "Observation space should be ({},), got {}".format(expected_obs_dim, obs_space.shape)
    
    # Test available actions
    key = jax.random.PRNGKey(0)
    observations, state = env.reset(key)
    avail_actions = env.get_avail_actions(state)
    
    for agent in env.agents:
        assert avail_actions[agent].tolist() == [1, 1, 1], \
            "All actions should be available, got {}".format(avail_actions[agent].tolist())
    
    print("> Test passed: Environment properties are correct")


def run_all_tests():
    """Run all SimpleCooperation tests."""
    print("Running SimpleCooperation Environment Tests")
    print("=" * 60)
    
    try:
        test_no_cooperation_full_episode()
        test_cooperation_immediate_termination()
        test_payoff_matrix()
        test_mixed_scenarios()
        test_environment_properties()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("> Episodes run for exactly max_steps when no mutual cooperation occurs")
        print("> Episodes end immediately when both agents cooperate")
        print("> Payoff matrix works correctly: [[1,0,0], [0,0,0], [0,0,0]]")
        print("> Termination logic works correctly in all scenarios")
        print("> Environment properties and interface are correct")
        
    except AssertionError as e:
        print("\nTEST FAILED: {}".format(e))
        raise
    except Exception as e:
        print("\nUNEXPECTED ERROR: {}".format(e))
        raise


if __name__ == "__main__":
    run_all_tests()

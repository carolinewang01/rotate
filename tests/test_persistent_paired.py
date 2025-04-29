import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from agents.agent_interface import ActorWithDoubleCriticPolicy
from agents.population_buffer import BufferedPopulation, PopulationBuffer
from open_ended_training.open_ended_persistent_paired import train_persistent_paired
from envs import make_env
from envs.log_wrapper import LogWrapper


def test_buffered_population():
    """Test basic functionality of BufferedPopulation class."""
    # Create a simple policy
    print("Creating policy")
    policy = ActorWithDoubleCriticPolicy(action_dim=5, obs_dim=10)
    
    # Create a population
    max_pop_size = 10
    print("Creating population")
    population = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=policy)
    
    # Generate some random parameters as example
    rng = jax.random.PRNGKey(42)
    example_params = policy.init_params(rng)
    
    # Initialize buffer
    print("Initializing buffer")
    buffer = population.reset_buffer(example_params)
    
    # Check initial state
    assert buffer.filled.sum() == 0, f"Buffer should have 0 filled agents, got {buffer.filled.sum()}"
    assert buffer.filled_count[0] == 0, f"Buffer should have 0 filled agents, got {buffer.filled_count[0]}"
    
    # Add an agent
    print("Adding agent")
    rng, rng1 = jax.random.split(rng)
    agent1_params = policy.init_params(rng1)
    buffer = population.add_agent(buffer, agent1_params, score=1.5)
    
    # Check updated state
    assert buffer.filled.sum() == 1, f"Buffer should have 1 filled agent, got {buffer.filled.sum()}"
    assert buffer.filled_count[0] == 1, f"Buffer should have 1 filled agent, got {buffer.filled_count[0]}"
    
    # Add another agent
    print("Adding another agent")
    rng, rng2 = jax.random.split(rng)
    agent2_params = policy.init_params(rng2)
    buffer = population.add_agent(buffer, agent2_params, score=2.0)
    
    # Check updated state
    assert buffer.filled.sum() == 2, f"Buffer should have 2 filled agents, got {buffer.filled.sum()}"
    assert buffer.filled_count[0] == 2, f"Buffer should have 2 filled agents, got {buffer.filled_count[0]}"
    
    # Sample from buffer
    print("Sampling from buffer")
    rng, sample_rng = jax.random.split(rng)
    # Sample only n=1 to make age assertion reliable
    n_samples = 1 
    indices, new_buffer = population.sample_agent_indices(buffer, n_samples, sample_rng)
    
    # Check that we got indices and buffer was updated
    assert indices.shape == (n_samples,), f"Indices shape should be ({n_samples},), got {indices.shape}"
    # Check that at least one age > 0. With n=1 sample, only one age is reset,
    # the other filled age (if present) should be incremented.
    assert jnp.any(new_buffer.ages > 0), f"At least one age should be > 0 after sampling n=1, got {new_buffer.ages}"
    
    print("BufferedPopulation tests passed!")
    return True

def test_persistent_paired():
    """Test the persistent PAIRED algorithm on a simple environment."""
    # Create test config
    config = {
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {},
        "PARTNER_POP_SIZE": 2,
        "NUM_CHECKPOINTS": 2,
        "NUM_OPEN_ENDED_ITERS": 2,
        "NUM_SEEDS": 1,
        "TRAIN_SEED": 42,
        "NUM_ENVS": 2,
        "ROLLOUT_LENGTH": 16,
        "NUM_UPDATES": 5,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 2,
        "LR": 1e-3,
        "ANNEAL_LR": False,
        "TIMESTEPS_PER_ITER_PARTNER": 320,  # 2*5*16*2
        "TIMESTEPS_PER_ITER_EGO": 320,
        "MAX_GRAD_NORM": 0.5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_POPULATION_SIZE": 10,
        "STALENESS_COEF": 0.3,
        "REPLAY_TEMP": 1.0,
        "CONF_BR_WEIGHT": 0.5,
        "NUM_EVAL_EPISODES": 2
    }
    
    # Create environment
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    # Generate random key
    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    
    # Run training
    print("Starting persistent PAIRED test...")
    final_ego_params, final_buffer, outs = train_persistent_paired(
        rng, env, config
    )
    
    # Check that the buffer has been populated
    filled_count = final_buffer.filled_count[0]
    print(f"Final population buffer has {filled_count} agents")
    assert filled_count > 0, "Population buffer should contain agents after training"
    
    # Extract ego agent metrics
    teammate_outs, ego_outs = outs
    
    # Verify some metrics exist
    assert "value_loss" in ego_outs["metrics"], "Missing value_loss in metrics"
    assert "actor_loss" in ego_outs["metrics"], "Missing actor_loss in metrics"
    
    print("Persistent PAIRED test completed successfully!")
    return True

if __name__ == "__main__":
    # Run tests
    # test_buffered_population()
    test_persistent_paired() 
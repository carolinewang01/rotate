import logging
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from agents.agent_interface import ActorWithDoubleCriticPolicy
from agents.population_buffer import BufferedPopulation, PopulationBuffer
from open_ended_training.open_ended_persistent_paired import train_persistent_paired
from envs import make_env
from envs.log_wrapper import LogWrapper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_buffered_population():
    """Test basic functionality of BufferedPopulation class."""
    # Create a simple policy
    policy = ActorWithDoubleCriticPolicy(action_dim=5, obs_dim=10)
    
    # Create a population
    max_pop_size = 10
    population = BufferedPopulation(max_pop_size=max_pop_size, policy_cls=policy)
    
    # Generate some random parameters as example
    rng = jax.random.PRNGKey(42)
    example_params = policy.init_params(rng)
    
    # Initialize buffer
    buffer = population.reset_buffer(example_params)
    
    # Check initial state
    assert buffer.filled.sum() == 0
    assert buffer.filled_count[0] == 0
    
    # Add an agent
    rng, rng1 = jax.random.split(rng)
    agent1_params = policy.init_params(rng1)
    buffer = population.add_agent(buffer, agent1_params, score=1.5)
    
    # Check updated state
    assert buffer.filled.sum() == 1
    assert buffer.filled_count[0] == 1
    
    # Add another agent
    rng, rng2 = jax.random.split(rng)
    agent2_params = policy.init_params(rng2)
    buffer = population.add_agent(buffer, agent2_params, score=2.0)
    
    # Check updated state
    assert buffer.filled.sum() == 2
    assert buffer.filled_count[0] == 2
    
    # Sample from buffer
    rng, sample_rng = jax.random.split(rng)
    indices, new_buffer = population.sample_agent_indices(buffer, 5, sample_rng)
    
    # Check that we got indices and buffer was updated
    assert indices.shape == (5,)
    assert new_buffer.ages[0] > 0 or new_buffer.ages[1] > 0  # At least one age was incremented
    
    log.info("BufferedPopulation tests passed!")
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
    
    # Initialize partner policy
    partner_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0]
    )
    
    # Create partner population
    partner_population = BufferedPopulation(
        max_pop_size=config["MAX_POPULATION_SIZE"],
        policy_cls=partner_policy,
        staleness_coef=config["STALENESS_COEF"],
        temp=config["REPLAY_TEMP"]
    )

    # Generate random key
    rng = jax.random.PRNGKey(config["TRAIN_SEED"])
    
    # Run training
    log.info("Starting persistent PAIRED test...")
    final_ego_params, final_buffer, outs = train_persistent_paired(
        rng, env, config, partner_policy, partner_population
    )
    
    # Check that the buffer has been populated
    filled_count = final_buffer.filled_count[0]
    log.info(f"Final population buffer has {filled_count} agents")
    assert filled_count > 0, "Population buffer should contain agents after training"
    
    # Extract ego agent metrics
    teammate_outs, ego_outs = outs
    
    # Verify some metrics exist
    assert "value_loss" in ego_outs["metrics"], "Missing value_loss in metrics"
    assert "actor_loss" in ego_outs["metrics"], "Missing actor_loss in metrics"
    
    log.info("Persistent PAIRED test completed successfully!")
    return True

if __name__ == "__main__":
    # Run tests
    test_buffered_population()
    test_persistent_paired() 
import shutil
import time
import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy
from agents.population_interface import AgentPopulation
from agents.population_buffer import BufferedPopulation, PopulationBuffer
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.run_episodes import run_episodes
from common.ppo_utils import Transition, unbatchify
from envs import make_env
from envs.log_wrapper import LogWrapper
from ego_agent_training.ppo_ego_with_buffer import train_ppo_ego_agent_with_buffer

# Import core functionality from original PAIRED implementation
from open_ended_training.open_ended_paired import train_regret_maximizing_partners, log_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def persistent_open_ended_training_step(carry, ego_policy, partner_population, config, env):
    '''
    Train the ego agent against a growing population of regret-maximizing partners.
    Unlike the original implementation, the partner population persists across iterations.
    '''
    prev_ego_params, population_buffer, rng = carry
    rng, partner_rng, ego_rng = jax.random.split(rng, 3)
    
    # Train partner agents with ego_policy
    train_out = train_regret_maximizing_partners(config, prev_ego_params, ego_policy, env, partner_rng)
    train_partner_params = train_out["checkpoints_conf"]
    
    # Add all checkpoints of each partner to the population buffer
    pop_size = config["PARTNER_POP_SIZE"]
    ckpt_size = config["NUM_CHECKPOINTS"]
    
    # Reshape parameters to flatten population and checkpoints dimensions
    # train_partner_params shape: (pop_size, ckpt_size, ...)
    # We need to reshape to (pop_size * ckpt_size, ...)
    def flatten_params(params):
        param_shape = params.shape[2:]  # shape after pop_size and ckpt_size
        # Reshape to combine pop_size and ckpt_size
        return params.reshape(pop_size * ckpt_size, *param_shape)
    
    flattened_params = jax.tree_map(flatten_params, train_partner_params)
    
    # Helper function to add each partner checkpoint to the buffer
    def add_partners_to_buffer(buffer, params_batch):
        def add_single_partner(carry_buffer, params):
            return partner_population.add_agent(carry_buffer, params), None
        
        new_buffer, _ = jax.lax.scan(
            add_single_partner,
            buffer,
            params_batch
        )
        return new_buffer
    
    # Add all checkpoints of all partners to the buffer
    updated_buffer = add_partners_to_buffer(population_buffer, flattened_params)
    
    # Train ego agent using the population buffer
    # Sample agents from buffer for training
    config["TOTAL_TIMESTEPS"] = config["TIMESTEPS_PER_ITER_EGO"]
    ego_out = train_ppo_ego_agent_with_buffer(
        config=config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        population_buffer=updated_buffer  # Pass the buffer to the training function
    )
    
    updated_ego_parameters = ego_out["final_params"]
    # Remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree_map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_buffer, rng)
    return carry, (train_out, ego_out)


def train_persistent_paired(rng, env, algorithm_config, partner_policy, partner_population):
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    # Initialize ego agent
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    
    # Initialize an empty population buffer
    population_buffer = partner_population.reset_buffer(init_ego_params)
    
    @jax.jit
    def open_ended_step_fn(carry, unused):
        return persistent_open_ended_training_step(carry, ego_policy, partner_population, algorithm_config, env)
    
    init_carry = (init_ego_params, population_buffer, rng)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    
    final_ego_params, final_buffer, _ = final_carry
    return final_ego_params, final_buffer, outs


def run_persistent_paired(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    # Initialize partner policy once - reused for all iterations
    partner_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0]
    )
    
    # Create persistent partner population with BufferedPopulation
    # The max_pop_size should be large enough to hold all agents across all iterations
    # Now we need more space since we're storing all checkpoints
    max_pop_size = algorithm_config.get("MAX_POPULATION_SIZE", 
                                      algorithm_config["PARTNER_POP_SIZE"] * 
                                      algorithm_config["NUM_CHECKPOINTS"] *
                                      algorithm_config["NUM_OPEN_ENDED_ITERS"] * 2)  # Add extra buffer space
    
    partner_population = BufferedPopulation(
        max_pop_size=max_pop_size,
        policy_cls=partner_policy,
        staleness_coef=algorithm_config.get("STALENESS_COEF", 0.3),
        temp=algorithm_config.get("REPLAY_TEMP", 1.0)
    )

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    log.info("Starting persistent open-ended PAIRED training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_persistent_paired, 
                env=env, algorithm_config=algorithm_config, 
                partner_policy=partner_policy, 
                partner_population=partner_population
                )
            )
        )
        all_ego_params, all_buffers, outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Persistent open-ended PAIRED training completed in {end_time - start_time} seconds.")

    # Log metrics (reusing the original PAIRED logging function)
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, wandb_logger, outs, metric_names)

    # Prepare return values for heldout evaluation
    _, ego_outs = outs
    ego_params = jax.tree_map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    return ego_policy, ego_params, init_ego_params, all_buffers 
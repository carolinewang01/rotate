'''
An implementation of open-ended FCP where both the ego agent and all 
partner agents are MLP actor critics.
'''
import shutil
import time
import logging
import jax
import numpy as np
import hydra
from functools import partial
from envs.log_wrapper import LogWrapper

from envs import make_env
from agents.agent_interface import MLPActorCriticPolicy, AgentPopulation
from agents.initialize_agents import initialize_s5_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from ppo.ippo import make_train as make_ppo_train
from ego_agent_training.ppo_ego import train_ppo_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_partners_in_parallel(config, partner_rng, env):
    '''
    Train a pool of partners for FCP using IPPO w/parameter sharing. 
    Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
    '''
    config["TOTAL_TIMESTEPS"] = config["TIMESTEPS_PER_ITER_PARTNER"] // config["PARTNER_POP_SIZE"]
    rngs = jax.random.split(partner_rng, config["PARTNER_POP_SIZE"])
    train_jit = jax.jit(jax.vmap(make_ppo_train(config, env)))
    out = train_jit(rngs)
    return out

def open_ended_training_step(carry, ego_policy, partner_population, config, env):
    prev_ego_params, rng = carry
    rng, partner_rng, train_ego_rng = jax.random.split(rng, 3)
    
    # Train partner agents
    train_partner_out = train_partners_in_parallel(config, partner_rng, env)
    
    # Extract partner parameters
    partner_params = train_partner_out["checkpoints"]
    pop_size = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]
    
    # Flatten partner parameters for AgentPopulation
    flattened_partner_params = jax.tree.map(
        lambda x: x.reshape((pop_size,) + x.shape[2:]), 
        partner_params
    )
    
    # Train ego agent using train_ppo_ego_agent
    config["TOTAL_TIMESTEPS"] = config["TIMESTEPS_PER_ITER_EGO"]
    ego_out = train_ppo_ego_agent(
        config=config,
        env=env,
        train_rng=train_ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )
    
    updated_ego_parameters = ego_out["final_params"]
    # remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree.map(lambda x: x.squeeze(axis=0), updated_ego_parameters)


    carry = (updated_ego_parameters, rng)
    return carry, (train_partner_out, ego_out)

def train_fcp(rng, env, algorithm_config, partner_policy, partner_population):
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    
    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, partner_population, algorithm_config, env)
    
    init_carry = (init_ego_params, rng)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )
    return outs
    
def run_fcp(config, wandb_logger):
    '''
    Run the open-ended FCP training loop.
    '''
    algorithm_config = dict(config["algorithm"])
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # Initialize partner policy once - reused for all iterations
    partner_policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
    )

    # Create partner population using the provided partner policy
    partner_population = AgentPopulation(
        pop_size=algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"],
        policy_cls=partner_policy
    )

    log.info("Starting open-ended FCP training...")
    start_time = time.time()
    
    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_fcp, 
                env=env, algorithm_config=algorithm_config, 
                partner_policy=partner_policy, 
                partner_population=partner_population
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Open-ended FCP training completed in {end_time - start_time} seconds.")

    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)
    
    # Log metrics
    log.info("Logging metrics...")
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_train_metrics(config, wandb_logger, outs, metric_names, 
                      num_controlled_actors=algorithm_config["NUM_ENVS"])
    return ego_policy, ego_params, init_ego_params

def log_train_metrics(config, logger, outs, 
                      metric_names: tuple, num_controlled_actors: int
                      ):
    """Process training metrics and log them using the provided logger.
    
    Args:
        config: dict, the configuration
        outs: tuple, contains (train_partner_out, ego_out) for each iteration
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    # partner metrics is a pytree where each leaf has shape 
    # (num_seeds, n_oel_updates, n_partners_seeds, num_partner_updates, rollout_length, agents_per_env * num_envs)
    # ego metrics is a pytree where each leaf has shape 
    # (num_seeds, n_oel_updates, 1, num_ego_updates, rollout_length, num_envs)

    # The discrepancy in the last dim is because the partner training process 
    # logs metrics for both agents in the environment, but the ego training process only logs
    # metrics for the ego agent.
    partner_outs, ego_outs = outs
    partner_metrics = partner_outs["metrics"]
    ego_metrics = ego_outs["metrics"]
    
    num_seeds = partner_metrics["returned_episode_returns"].shape[0]
    num_open_ended_iters = ego_metrics["returned_episode_returns"].shape[1]
    num_partner_updates = partner_metrics["returned_episode_returns"].shape[3]
    num_ego_updates = ego_metrics["returned_episode_returns"].shape[3]

    # Extract partner train stats
    partner_metrics = jax.tree.map(lambda x: x[..., :num_controlled_actors], partner_metrics)
    partner_stats = get_stats(partner_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_partner_seeds, num_partner_updates, 2)
    partner_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], partner_stats) # shape (num_open_ended_iters, num_partner_updates)

    # Extract ego train stats
    ego_stats = get_stats(ego_metrics, metric_names) # shape (num_seeds, num_open_ended_iters, num_ego_seeds, num_ego_updates, 2)
    ego_stat_means = jax.tree.map(lambda x: np.mean(x, axis=(0, 2))[..., 0], ego_stats) # shape (num_open_ended_iters, num_ego_updates)

    # Average ego metrics
    # shape (num_seeds, num_open_ended_iters, n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    avg_ego_value_losses = np.asarray(ego_metrics["value_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_actor_losses = np.asarray(ego_metrics["actor_loss"]).mean(axis=(0, 2, 4, 5))
    avg_ego_entropy_losses = np.asarray(ego_metrics["entropy_loss"]).mean(axis=(0, 2, 4, 5))
    # Process ego eval return metrics
    # shape (num_seeds, num_open_ended_iters, n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, num_agents_per_env)
    avg_ego_rets_per_iter = np.asarray(ego_metrics["eval_ep_last_info"]["returned_episode_returns"]).mean(axis=(0, 2, 4, 5, 6))

    for iter_idx in range(num_open_ended_iters):
        # Log partner train stats
        for step in range(num_partner_updates):
            global_step = iter_idx * num_partner_updates + step
            for stat_name, stat_data in partner_stat_means.items():
                logger.log_item(f"Train/Partner_{stat_name}", stat_data[iter_idx, step], train_step=global_step)
        
        # Log metrics for each ego update step
        for step in range(num_ego_updates):
            # Include iteration in the step calculation
            global_step = iter_idx * num_ego_updates + step
            for stat_name, stat_data in ego_stat_means.items():
                logger.log_item(f"Train/Ego_{stat_name}", stat_data[iter_idx, step], train_step=global_step)
            
            logger.log_item("Eval/EgoReturn", avg_ego_rets_per_iter[iter_idx, step], train_step=global_step)
            logger.log_item("Losses/EgoValueLoss", avg_ego_value_losses[iter_idx, step], train_step=global_step)
            logger.log_item("Losses/EgoActorLoss", avg_ego_actor_losses[iter_idx, step], train_step=global_step)
            logger.log_item("Losses/EgoEntropyLoss", avg_ego_entropy_losses[iter_idx, step], train_step=global_step)
                
    logger.commit()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

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
from envs.log_wrapper import LogWrapper

from envs import make_env
from common.wandb_visualizations import Logger
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
    # (n_oel_updates, n_partners_seeds, n_updates, rollout_length, agents_per_env * num_envs)
    # ego metrics is a pytree where each leaf has shape 
    # (n_oel_updates, 1, n_updates, rollout_length, num_envs)

    # The discrepancy in the last dim is because the partner training process 
    # logs metrics for both agents in the environment, but the ego training process only logs
    # metrics for the ego agent.
    partner_outs, ego_outs = outs
    partner_metrics = partner_outs["metrics"]
    ego_metrics = ego_outs["metrics"]
    
    num_open_ended_iters = ego_metrics["returned_episode_returns"].shape[0]
    num_partner_updates = partner_metrics["returned_episode_returns"].shape[2]
    num_ego_updates = ego_metrics["returned_episode_returns"].shape[2]

    for iter_idx in range(num_open_ended_iters):
        ### partner metrics
        # Extract partner train stats
        partner_metrics_iter = jax.tree.map(lambda x: x[iter_idx, ..., :num_controlled_actors], partner_metrics)
        # get_stats expects a pytree of shape (n_seeds, n_updates, n_rollouts, n_envs)
        partner_stats = get_stats(partner_metrics_iter, metric_names)
        partner_stats = {k: np.mean(np.array(v), axis=0) for k, v in partner_stats.items()}
        
        for step in range(num_partner_updates):
            for stat_name, stat_data in partner_stats.items():
                stat_mean = stat_data[step, 0]
                # Include iteration in the step calculation
                global_step = iter_idx * num_partner_updates + step
                logger.log_item(f"Train/Partner_{stat_name}", stat_mean, train_step=global_step)
        
        ### ego metrics
        # Extract ego train stats
        ego_metrics_iter = jax.tree.map(lambda x: x[iter_idx], ego_metrics)
        ego_stats = get_stats(ego_metrics_iter, metric_names)
        ego_stats = {k: np.mean(np.array(v), axis=0) for k, v in ego_stats.items()}

        # Process value, actor, and entropy losses
        # initial shape is   # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
        average_ego_value_losses = np.asarray(ego_metrics_iter["value_loss"]).mean(axis=(0, 2, 3))
        average_ego_actor_losses = np.asarray(ego_metrics_iter["actor_loss"]).mean(axis=(0, 2, 3))
        average_ego_entropy_losses = np.asarray(ego_metrics_iter["entropy_loss"]).mean(axis=(0, 2, 3))
                
        # Process eval return metrics
        all_ego_returns = np.asarray(ego_metrics_iter["eval_ep_last_info"]["returned_episode_returns"])  # shape (n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, num_agents_per_env)
        average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0, 2, 3, 4))
                
        # Log metrics for each ego update step
        for step in range(num_ego_updates):
            # Include iteration in the step calculation
            global_step = iter_idx * num_ego_updates + step
            for stat_name, stat_data in ego_stats.items():
                if stat_data.shape[0] > step:  # Check if we have data for this step
                    stat_mean = stat_data[step, 0]
                    logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=global_step)
            
            logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[step], train_step=global_step)
            logger.log_item("Losses/EgoValueLoss", average_ego_value_losses[step], train_step=global_step)
            logger.log_item("Losses/EgoActorLoss", average_ego_actor_losses[step], train_step=global_step)
            logger.log_item("Losses/EgoEntropyLoss", average_ego_entropy_losses[step], train_step=global_step)
                
    logger.commit()
    
    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
    
def run_fcp(config, wandb_logger):
    '''
    Run the open-ended FCP training loop.
    '''
    algorithm_config = dict(config["algorithm"])
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, ego_init_rng, train_rng = jax.random.split(rng, 3)

    # Initialize ego agent
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, ego_init_rng)
    
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

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return open_ended_training_step(carry, ego_policy, partner_population, algorithm_config, env)
    
    # Define initial carry with policies included
    init_carry = (init_ego_params, train_rng)

    log.info("Starting open-ended FCP training...")
    start_time = time.time()
    with jax.disable_jit(False):
        final_carry, outs = jax.lax.scan(
            open_ended_step_fn, 
            init_carry, 
            xs=None,
            length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
        )
    
    end_time = time.time()
    log.info(f"Open-ended FCP training completed in {end_time - start_time} seconds.")

    # Prepare return values for heldout evaluation
    _ , ego_outs = outs
    ego_params = jax.tree.map(lambda x: x[:, 0], ego_outs["final_params"]) # shape (num_open_ended_iters, num_ego_seeds, num_ckpts, leaf_dim)
    
    # Log metrics
    log.info("Logging metrics...")
    metric_names = get_metric_names(config["ENV_NAME"])
    log_train_metrics(config, wandb_logger, outs, metric_names, 
                      num_controlled_actors=algorithm_config["NUM_ENVS"])
    return ego_policy, ego_params, init_ego_params

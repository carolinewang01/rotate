import shutil
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from agents.agent_interface import AgentPopulation, MLPActorCriticPolicy
from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from common.ppo_utils import Transition, unbatchify
from common.run_episodes import run_episodes
from envs import make_env
from envs.log_wrapper import LogWrapper
from ego_agent_training.ppo_ego import train_ppo_ego_agent
from evaluation.agent_loader_from_config import initialize_rl_agent_from_config
from evaluation.heldout_evaluator import load_heldout_set

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestAgentPopulation(AgentPopulation):
    ''' The main difference from the AgentPopulation is that 
    the test mode is a class attribute, so it remains static for the lifetime of the object
    '''
    def __init__(self, pop_size, policy_cls, test_mode=False):
        super().__init__(pop_size, policy_cls)
        self.test_mode = test_mode

    def get_actions(self, pop_params, agent_indices, obs, done, avail_actions, hstate, rng):
        '''
        Get the actions of the agents specified by agent_indices. Does not support agents that 
        require environment state or auxiliary observations.
        Returns:
            actions: actions with shape (num_envs,)
            new_hstate: new hidden state with shape (num_envs, ...) or None
        '''
        gathered_params = self.gather_agent_params(pop_params, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                              aux_obs=None, 
                                              env_state=None, 
                                              test_mode=self.test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate


def run_br_training(config, wandb_logger):
    '''Run ego agent training against the population of partner agents.
    
    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    partner_agent_config = dict(config["partner_agent"])
    heldout_agents = load_heldout_set(partner_agent_config, env, config["TASK_NAME"], config["ENV_KWARGS"], init_rng)
    assert len(heldout_agents) == 1, "Only supports training against one partner agent for now."

    partner_name = list(heldout_agents.keys())[0]
    partner_policy, partner_params, partner_test_mode = list(heldout_agents.values())[0]
    
    # Create partner population
    partner_population = AgentPopulation(
        pop_size=1,
        policy_cls=partner_policy,
        test_mode=partner_test_mode
    )
    
    # Initialize ego agent
    if algorithm_config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif algorithm_config["EGO_ACTOR_TYPE"] == "rnn":
        # WARNING: currently the RNN policy is not working. 
        # TODO: fix this!
        ego_policy, init_ego_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    
    log.info("Starting ego agent training...")
    start_time = time.time()
    
    # Run the training
    out = train_ppo_ego_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=partner_params
    )
    
    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    # log_metrics(config, out, wandb_logger, metric_names)
    
    return out
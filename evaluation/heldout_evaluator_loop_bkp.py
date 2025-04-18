'''This script implements evaluating ego agents against heldout agents. 
Warning: ActorCritic agents that rely on auxiliary information to compute actions are not currently supported.'''

import jax

from agents.lbf.agent_policy_wrappers import LBFRandomPolicyWrapper
from evaluation.agent_loader_from_config import initialize_rl_agent_from_config
from common.run_episodes import run_episodes
from common.tree_utils import tree_stack
from common.plot_utils import get_metric_names
from envs import make_env
from envs.log_wrapper import LogWrapper
from prettytable import PrettyTable

def extract_params(params, init_params, idx_list=None):
    '''params is a pytree of n model checkpoints, where each leaf has an unknown number 
    of checkpoint dimensions, and the last dimension corresponds to the layer dimension. 
    This function extracts each of the n checkpoints and returns a list of n pytrees, 
    where each pytree has the same structure as init_params.

    Args:
        params: pytree of n checkpoints (n >= 1)
        init_params: pytree corresp. to ONE checkpoint. used as a reference for the structure of the output pytrees.
        idx_list: list of indices that each checkpoint corresponds to. If None, all checkpoints will be extracted.
    Returns:
        list of n pytrees with same structure as init_params
    '''
    assert jax.tree.structure(params) == jax.tree.structure(init_params), "Params and init_params must have the same structure."

    model_list = []
    params_shape = jax.tree.leaves(params)[0].shape
    init_params_shape = jax.tree.leaves(init_params)[0].shape
    
    # only one model, no extraction needed
    if params_shape == init_params_shape:
        model_list = [params]
    # multiple models, extract each one
    else:
        # first, flatten the params so that each leaf has shape (..., init_params_shape)
        flattened_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), params, init_params)        
        # then, extract each model
        n_models = jax.tree.leaves(flattened_params)[0].shape[0]
        for i in range(n_models):
            model_i = jax.tree.map(lambda x: x[i], flattened_params)
            model_list.append(model_i)
    return model_list

def load_heldout_set(heldout_config, env, env_name, rng):
    '''Load heldout evaluation agents from config.
    Returns a dictionary of agents with keys as agent names and values as tuples of (policy, params).
    '''
    heldout_agents = {}
    for agent_name, agent_config in heldout_config.items():
        params_list = None
        # Load RL-based agents
        if "path" in agent_config:
            # ensure that each rl agent has a unique initialization rng
            rng, init_rng = jax.random.split(rng)
            policy, params, init_params = initialize_rl_agent_from_config(agent_config, agent_name, env, init_rng)
            # params contains multiple model checkpoints, so we need to extract each one
            params_list = extract_params(params, init_params, agent_config["idx_list"])

        # Load non-RL-based heuristic agents
        elif env_name == 'lbf':
            # load heuristic agents
            if agent_config["actor_type"] == 'random_agent':
                policy = LBFRandomPolicyWrapper()
        elif env_name == 'overcooked-v1':
            pass # TODO: implement overcooked agent policy wrappers
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        if params_list is None: # heuristic agent
            heldout_agents[agent_name] = (policy, None)
        else: # rl agent
            for i, params_i in enumerate(params_list):
                heldout_agents[f'{agent_name}_{i}'] = (policy, params_i)

    return heldout_agents


def eval_egos_vs_heldouts(config, env, rng, num_episodes, ego_agent_list, heldout_agent_list):
    '''Evaluate all ego agents against all heldout partners.
    Args: 
        ego_agent_list: a list of (policy, params) tuples for each ego agent
        heldout_partners: a list of (policy, params) tuples for each heldout partner. params might be None for heuristic agents.
    Returns a pytree of shape (num_ego_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
    '''
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    num_ego_agents = len(ego_agent_list)
    num_partner_total = len(heldout_agent_list)

    def eval_pair_fn(rng, ego_policy, ego_param, partner_policy, partner_param):
        return run_episodes(rng, env, 
                            agent_0_param=ego_param, agent_0_policy=ego_policy,
                            agent_1_param=partner_param, agent_1_policy=partner_policy,
                            max_episode_steps=config["MAX_EPISODE_STEPS"],
                            num_eps=num_episodes)

    # Initialize results array
    eval_metrics = []
    
    # Split RNG for each ego agent
    rngs = jax.random.split(rng, num_ego_agents)
    
    # Simple double for loop implementation
    for ego_idx in range(num_ego_agents):
        ego_agent = ego_agent_list[ego_idx]
        ego_policy, ego_param = ego_agent
        ego_rng = rngs[ego_idx]
        
        # Split RNG for each heldout partner
        partner_rngs = jax.random.split(ego_rng, num_partner_total)
        
        ego_metrics = []
        for partner_idx in range(num_partner_total):
            heldout_agent = heldout_agent_list[partner_idx]
            partner_policy, partner_param = heldout_agent
            partner_rng = partner_rngs[partner_idx]
            
            # Evaluate the pair
            metrics = eval_pair_fn(partner_rng, ego_policy, ego_param, partner_policy, partner_param)
            ego_metrics.append(metrics)

        eval_metrics.append(tree_stack(ego_metrics))    
    return tree_stack(eval_metrics)


def run_heldout_evaluation(config, print_metrics=False):
    '''Run heldout evaluation'''
    # wandb_logger = Logger(config)

    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["EVAL_SEED"])
    rng, ego_init_rng, heldout_init_rng, eval_rng = jax.random.split(rng, 4)
    
    # load ego agents
    ego_agent_config = dict(config["ego_agent"])
    ego_policy, ego_params, init_ego_params = initialize_rl_agent_from_config(ego_agent_config, "ego", env, ego_init_rng)
    ego_params_list = extract_params(ego_params, init_ego_params, ego_agent_config["idx_list"])
    ego_agent_list = [(ego_policy, p) for p in ego_params_list]
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["ENV_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["ENV_NAME"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(config, env, eval_rng, config["NUM_EVAL_EPISODES"], ego_agent_list, heldout_agent_list)

    if print_metrics:
        # each leaf of eval_metrics has shape (num_ego_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
        metric_names = get_metric_names(config["ENV_NAME"])
        ego_names = [f"ego_{i}" for i in range(len(ego_agent_list))]
        heldout_names = list(heldout_agents.keys())
        for metric_name in metric_names:
            print_metrics_table(eval_metrics, metric_name, ego_names, heldout_names)
    return eval_metrics

def print_metrics_table(eval_metrics, metric_name, ego_names, heldout_names):
    '''Print a table of the mean and std of the metric for each ego agent and heldout agent.'''
    metric_data_mean = eval_metrics[metric_name][..., 0].mean(axis=-1) # shape (num_ego_agents, num_heldout_agents)
    metric_data_std = eval_metrics[metric_name][..., 0].std(axis=-1) # shape (num_ego_agents, num_heldout_agents)
    table = PrettyTable()
    table.field_names = ["---", *heldout_names]
    for i, ego_name in enumerate(ego_names):
        row = [ego_name] + [f"{metric_data_mean[i, j]:.2f} ± {metric_data_std[i, j]:.2f}" for j in range(len(heldout_names))]
        table.add_row(row)
    print(f"\n{metric_name} (mean ± std):")
    print(table)

import os
import jax
import hydra

from agents.lbf.agent_policy_wrappers import LBFRandomPolicyWrapper
from common.save_load_utils import load_train_run
from common.wandb_visualizations import Logger
from evaluation.agent_loader_from_config import initialize_rl_agent_from_config
from envs import make_env
from envs.log_wrapper import LogWrapper

def extract_params(params, init_params):
    '''params is a pytree of n model checkpoints, where each leaf has an unknown number 
    of checkpoint dimensions, and the last dimension corresponds to the layer dimension. 
    This function extracts each of the n checkpoints and returns a list of n pytrees, 
    where each pytree has the same structure as init_params.

    Args:
        params: pytree of n checkpoints (n >= 1)
        init_params: pytree corresp. to ONE checkpoint. used as a reference for the structure of the output pytrees.

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
        try:
            flattened_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), params, init_params)
            print("Successfully flattened params")

        except Exception as e:
            print(f"Error flattening params: {e}")
            import pdb; pdb.set_trace()
        
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
            policy, params, init_params = initialize_rl_agent_from_config(agent_config, env, init_rng)
            # params contains multiple model checkpoints, so we need to extract each one
            params_list = extract_params(params, init_params)

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

def run_heldout_evaluation(config):
    '''Run heldout evaluation'''
    # wandb_logger = Logger(config)

    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["EVAL_SEED"])
    rng, ego_init_rng, heldout_init_rng, eval_rng = jax.random.split(rng, 4)
    # ego_agent_config = dict(config["ego_agent"])
    # ego_policy, ego_params, init_params = initialize_rl_agent(ego_agent_config, env, ego_init_rng)
    heldout_cfg = config["heldout_set"][config["ENV_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["ENV_NAME"], heldout_init_rng)

    import pdb; pdb.set_trace()

    print("Done")

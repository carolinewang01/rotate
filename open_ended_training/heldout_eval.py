import jax
import numpy as np
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import eval_egos_vs_heldouts, load_heldout_set


# TODO: figure out a better place to put this script
def run_heldout_evaluation(config, ego_policy, ego_params, init_ego_params):
    '''Run heldout evaluation given an ego policy, ego params, and init_ego_params.'''
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(config["EVAL_SEED"])
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)
    
    # flatten ego checkpoints and idx labels
    flattened_ego_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), ego_params, init_ego_params)      
    num_ego_agents = jax.tree.leaves(flattened_ego_params)[0].shape[0]
    ego_names = [f"ego ({i})" for i in range(num_ego_agents)]
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["ENV_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["ENV_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    # run evaluation
    eval_metrics = eval_egos_vs_heldouts(config, env, eval_rng, config["NUM_EVAL_EPISODES"], 
                                        ego_policy, flattened_ego_params, heldout_agent_list)
    
    return eval_metrics, ego_names, heldout_names
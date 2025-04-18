import logging
import jax
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_actor_with_double_critic, \
    initialize_actor_with_conditional_critic
from common.save_load_utils import load_checkpoints

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def initialize_rl_agent_from_config(agent_config, env, rng):
    '''Load RL agent from checkpoint and initialize from config.
    The agent_config dictionary should have the following structure:
    {
        "path": str,
        "actor_type": str,
        "ckpt_key": str, # key to load from checkpoint. Default is "checkpoints".
        "idx_list": list, # list of indices to load from checkpoint. If null, all checkpoints will be loaded.
        # and any other parameters needed to initialize the agent policy
    }
    '''
    assert "path" in agent_config, "Path to agent checkpoint must be provided."
    assert "actor_type" in agent_config, "Actor type must be provided."
    assert "idx_list" in agent_config, "Indices to load from checkpoint must be provided."

    agent_path = agent_config["path"]
    ckpt_key = agent_config.get("ckpt_key", "checkpoints")
    agent_ckpt = load_checkpoints(agent_path, ckpt_key=ckpt_key)

    leaf0_shape = jax.tree.leaves(agent_ckpt)[0].shape

    if agent_config["idx_list"] is None: # load all checkpoints
        idx_list = [slice(None)]
    else: # load specific checkpoints
        # convert omegaconf list config to list recursively
        idx_list = OmegaConf.to_object(agent_config["idx_list"])
        idx_list = jax.tree.map(lambda x: int(x), idx_list)
        if len(idx_list) == 1: # special handling needed for single index
            idx_tuple = tuple(idx_list[0])
        else:
            idx_tuple = tuple(idx_list)
    
    log.info(f"Loaded agent checkpoint where leaf 0 has shape {leaf0_shape}. "
             f" Selecting indices {idx_list} for evaluation.")

    agent_params = jax.tree.map(lambda x: x[idx_tuple], agent_ckpt)

    rng, init_rng, train_rng = jax.random.split(rng, 3)
    
    if agent_config["actor_type"] == "s5":
        policy, init_params = initialize_s5_agent(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "mlp":
        policy, init_params = initialize_mlp_agent(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "rnn":
        policy, init_params = initialize_rnn_agent(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "actor_double_critic":
        policy, init_params = initialize_actor_with_double_critic(agent_config, env, init_rng)
    elif agent_config["actor_type"] == "actor_with_conditional_critic":
        policy, init_params = initialize_actor_with_conditional_critic(agent_config, env, init_rng)
    else:
        raise ValueError(f"Invalid actor type: {agent_config['actor_type']}")

    return policy, agent_params, init_params

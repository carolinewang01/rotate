import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper

from envs import make_env
from ppo.ippo import unbatchify
from common.mlp_actor_critic import ActorWithConditionalCritic
from common.wandb_visualizations import Logger
from typing import NamedTuple

class XPTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    self_id: jnp.ndarray
    oppo_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def linear_schedule(config, count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

def train(env, config, rng):
    # initialize confederate
    conf_agent_net = ActorWithConditionalCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    # initialize best response
    br_agent_net = ActorWithConditionalCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    
    rng, init_conf_rng, init_br_rng = jax.random.split(rng, 3)
    all_conf_init_rngs = jax.random.split(init_conf_rng, config["POPULATION_SIZE"])
    all_br_init_rngs = jax.random.split(init_br_rng, config["POPULATION_SIZE"])

    def init_single_pair_optimizers(rng_agent, rng_br):

        # Initialize parameters of the generated confederate and BR policy
        init_x = ( # init obs, avail_actions
            jnp.zeros(env.observation_space(env.agents[0]).shape),
            jnp.zeros(config["POPULATION_SIZE"]),
            jnp.ones(env.action_space(env.agents[0]).n),
        )
        init_params = conf_agent_net.init(rng_agent, init_x)

        init_x = ( # init obs, avail_actions
            jnp.zeros(env.observation_space(env.agents[0]).shape),
            jnp.zeros(config["POPULATION_SIZE"]),
            jnp.ones(env.action_space(env.agents[0]).n),
        )
        init_params_br = br_agent_net.init(rng_br, init_x)

        # Define optimizers for both confederate and BR policy
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_br = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_br = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state_conf = TrainState.create(
            apply_fn=conf_agent_net.apply,
            params=init_params,
            tx=tx,
        )

        train_state_br = TrainState.create(
            apply_fn=br_agent_net.apply,
            params=init_params_br,
            tx=tx_br,
        )

        return train_state_conf, train_state_br
    
    init_all_networks_and_optimizers = jax.vmap(init_single_pair_optimizers)
    all_conf_optims, all_br_optims = init_all_networks_and_optimizers(all_conf_init_rngs, all_br_init_rngs)
    jax.debug.breakpoint()

if __name__ == "__main__":
    env = make_env("lbf")
    config = {
        "NUM_MINIBATCHES": 4,
        "ALG": "paired",
        "NUM_OPEN_ENDED_ITERS": 1,
        "PARTNER_POP_SIZE": 10,
        "TRAIN_SEED": 38410,
        "NUM_ENVS": 16,
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {},
        "ROLLOUT_LENGTH": 128,
        "TOTAL_TIMESTEPS": 1.6e7, # one step of open-ended training
        "LR": 1.e-4,
        "UPDATE_EPOCHS": 15,
        "NUM_MINIBATCHES": 4,
        "MAX_EVAL_EPISODES": 20,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.001,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "POPULATION_SIZE": 5
    }

    num_agents=2
    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

    # Right now assume control of both agent and its BR
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"])// config["NUM_ENVS"]
    config["MINIBATCH_SIZE_EGO"] = ((config["NUM_ACTORS"]-1) * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]
    config["MINIBATCH_SIZE_BR"] = (config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

    train(env, config, jax.random.PRNGKey(100))
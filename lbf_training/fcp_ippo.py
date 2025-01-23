"""
Based on PureJaxRL Implementation of PPO. 
Script adapted from JaxMARL IPPO RNN Smax script.
"""
import jax
import jax.numpy as jnp
import functools
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import jumanji
import wandb
import pickle

from lbf_training.ippo_lbf_checkpoints import make_train

# def train_one_seed(rng: jax.random.PRNGKey, config, num_checkpoints=5):
#     train_fn = make_train(config)
#     out = train_fn(rng)
#     final_train_state = out["runner_state"][0]
#     final_params = final_train_state.params
    
#     # Example: store the final params multiple times as “checkpoints”
#     checkpoints = []
#     for _ in range(num_checkpoints):
#         checkpoints.append(flax.serialization.to_bytes(final_params))

#     # Return as an object array so that JAX doesn't try to treat them as numeric
#     return {"checkpoints": jnp.array(checkpoints, dtype=object), "metrics": out["metrics"]}

def train_partners_in_parallel(config):
    '''
    out is a dictionary with keys runner_state, metrics, and checkpoints.
    '''
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rngs, config["NUM_SEEDS"])

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config)))
        out = train_jit(rngs)
    
    return out['checkpoints'], out['metrics']

def save_checkpoints(config, checkpoints):
    with open(f"checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoints, f)

if __name__ == "__main__":
    # set hyperparameters:
    config = {
        "LR": 1.e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16, # 4,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.01,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {
        # "layout" : "cramped_room"
        },
        "ANNEAL_LR": True,
        "SEED": 0,
        "NUM_SEEDS": 3
    }

    out = train_partners_in_parallel(config)

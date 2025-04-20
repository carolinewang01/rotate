import time
import logging

import jax

from envs import make_env
from envs.log_wrapper import LogWrapper
from ppo.ippo import make_train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_partners_in_parallel(config, base_seed):
    '''
    Train a pool of partners for FCP. Return checkpoints for all partners.
    Returns out, a dictionary of the final train_state, metrics, and checkpoints.
    '''
    start_time = time.time()
    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    debug_mode = False
    with jax.disable_jit(debug_mode):
        if debug_mode: 
            out = make_train(config)(rngs)
        else:
            train_jit = jax.jit(jax.vmap(make_train(config, env)))
            out = train_jit(rngs)
    end_time = time.time()
    log.info(f"Training partners took {end_time - start_time:.2f} seconds.")
    return out

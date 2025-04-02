import os
import pickle
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np


def save_train_run(out, savedir, savename):
    '''Save train run as orbax checkpoint'''
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, savename)
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(out)
    
    # Save the checkpoint
    checkpointer.save(savepath, out, save_args=save_args)
    return savepath

def load_checkpoints(path):
    '''Load checkpoints from orbax checkpoint
    '''
    # load the checkpoint
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(path)
    # convert pytree leaves from np arrays to jax arrays
    restored = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        restored
    )
    return restored

def save_train_run_as_pickle(out, savedir, savename):
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    savepath = f"{savedir}/{savename}.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(out, f)
    return savepath

def load_checkpoints_from_pickle(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
        checkpoints = out["checkpoints"]
    return checkpoints
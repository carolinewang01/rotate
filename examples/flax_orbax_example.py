import os
import shutil

import numpy as np
from flax import linen as nn
from flax.training import train_state
import jax
from jax import random, numpy as jnp

import orbax.checkpoint
from flax.training import orbax_utils
import optax

ckpt_dir = 'results/flax_orbax_example/'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

#################### CREATE STATE
# A simple model with one linear layer.
key1, key2 = random.split(random.key(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)
# Perform a simple gradient update similar to the one during a normal training workflow.
state = state.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, state.params))

# Some arbitrary nested pytree with a dictionary and a NumPy array.
config = {'dimensions': np.array([5, 3])}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}

# Print original types
print("\nOriginal types:")
print("x1 type:", type(x1))
print("x1 dtype:", x1.dtype)
print("model params types:")
for k, v in state.params.items():
    print(f"  {k}: {type(v)}")
    print(f"  {k} dtype: {v.dtype}")

#################### SAVE CHECKPOINT

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
print("\nSave args are: ", save_args)
orbax_checkpointer.save(ckpt_dir, ckpt, save_args=save_args)

#################### RESTORE CHECKPOINT

restored = orbax_checkpointer.restore(ckpt_dir)
print("\nRestored checkpoint structure:", restored.keys())

# Print restored types
print("\nRestored types:")
print("restored['data'][0] type:", type(restored['data'][0]))
print("restored['data'][0] dtype:", restored['data'][0].dtype)
print("restored model params types:")
for k, v in restored['model']['params'].items():
    print(f"  {k}: {type(v)}")
    print(f"  {k} dtype: {v.dtype}")

#################### PERFORM FORWARD PASS

# Convert restored params to JAX arrays if needed
restored_params = jax.tree_util.tree_map(
    lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, 
    restored['model']['params']
)

# Define a simple input
x2 = random.normal(random.key(1), (5,))

# Perform a forward pass with the restored model
y2 = model.apply({'params': restored_params}, x2)
print("\nForward pass result:", y2)

# Verify the output type
print("Output type:", type(y2))
print("Output dtype:", y2.dtype)

# Compare with original model output
y1 = model.apply({'params': state.params}, x1)
print("\nOriginal model output:", y1)
print("Original output type:", type(y1))
print("Original output dtype:", y1.dtype)

# Compare outputs numerically (using different inputs, so outputs will be different)
print("\nOutputs are different as expected since we used different inputs (x1 vs x2)")
print("But both outputs are valid JAX arrays with the correct shape")
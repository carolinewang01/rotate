import jax
import jax.numpy as jnp
import numpy as np

######## Test 1: 1d indexing
# a = jnp.zeros((5, 4))
# b = jnp.zeros((5, 4))
# tree = (a, b)

# idx_list = [0, 1, 2] # desired shape (3,)
# idxs = np.array(idx_list)
# print("idxs shape", idxs.shape)

# sliced_tree = jax.tree.map(lambda x: x[idxs], tree)
# print(sliced_tree[0].shape)

# import sys; sys.exit(0)

######## Test 2: 2d indexing
a = jnp.zeros((5, 5, 4))
b = jnp.zeros((5, 5, 4))

tree = (a, b)

# idx_list = [[0, 0], [0, 1]] # desired shaped from slicing (2, 4) - this works!
# idx_list = [[0, 0]] # desired shaped from slicing: (4,) or (1, 4) - this doesn't work!
# idx_list = [0, 0] # shape is (4,)
# idx_list = [slice(None)]

idx_list = [[0, 0], [0, 1], [0, 2]] # we expect (3, 4), but instead get (2,)

idxs = np.array(idx_list)
print("idxs shape", idxs.shape)
rows = idxs[:, 0]
cols = idxs[:, 1]
print("rows", rows)
print("cols", cols)

sliced_tree = jax.tree.map(lambda x: x[(rows, cols)], tree)

print(sliced_tree[0].shape)


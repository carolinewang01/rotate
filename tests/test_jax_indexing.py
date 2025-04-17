import jax
import jax.numpy as jnp

a = jnp.zeros((3, 3, 4))
b = jnp.zeros((3, 3, 4))

tree = (a, b)

# idx_list = [[0, 0], [0, 1]] # desired shaped from slicing (2, 4) - this works!
# idx_list = [[0, 0]] # desired shaped from slicing: (4,) or (1, 4) - this doesn't work!
# idx_list = [0, 0] # shape is (4,)
idx_list = [slice(None)]
idx_tuple = tuple(idx_list)
print(idx_tuple)

sliced_tree = jax.tree.map(lambda x: x[idx_tuple], tree)

print(sliced_tree[0].shape)


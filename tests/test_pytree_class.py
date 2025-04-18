import sys
import jax
import jax.numpy as jnp
from common.tree_utils import tree_stack, tree_unstack

# test combining two pytrees into a single pytree
atree = {"a": jnp.zeros((4,)), "b": jnp.zeros((4,))}
b = [atree, atree]
c = tree_stack(b)
print(jax.tree.structure(b))
print(jax.tree.structure(c))

sys.exit(0)

################
# test flattening a pytree based on the structure of another pytree
atree = {"a": jnp.zeros((1, 2, 16)), "b": jnp.zeros((1, 2, 16))}
btree = {"a": jnp.zeros((16,)), "b": jnp.zeros((16,))}

flattened_atree = jax.tree.map(lambda x, y: x.reshape( (-1,)+ y.shape), atree, btree)

print(jax.tree.leaves(flattened_atree)[0].shape)
print(jax.tree.structure(flattened_atree))

sys.exit(0)

###############
'''Testing whether python classes can be used as pytree leaves.'''
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Dog(name={self.name}, age={self.age})"
    
    def get_num_legs(self):
        return 4
    
fido = Dog("Fido", 3)
spot = Dog("Spot", 2)
carl = Dog("Carl", 4)

atree = [(fido, 1), (spot, 2), (carl, 3)]
print(jax.tree.leaves(atree))
print(jax.tree.structure(atree))
# print(jax.tree.map(lambda x: x.get_num_legs(), atree))

#############
btree = {"a": fido, "b": spot, "c": carl}
print(jax.tree.leaves(btree))
print(jax.tree.structure(btree))




    
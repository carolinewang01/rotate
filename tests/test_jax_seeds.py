import jax

# create initial rng
rng = jax.random.PRNGKey(38410)
print("initial rng: ", rng)

rngs = jax.random.split(rng, 4)
print("rngs: ", rngs)

# try to reshape rngs into a 2D array
rngs = rngs.reshape(2, 2, 2)
print("rngs reshaped into 2x2 array: ", rngs)
import sys; sys.exit(0)

###################
# split rng into two
_, rng1 = jax.random.split(rng)
print("rng1, split from initial rng: ", rng1)

# show that rng is deterministic
_, rng2 = jax.random.split(rng)
print("rng2, split from initial rng: ", rng2)

# show that rng1 and rng2 are the same
print("rng1 == rng2: ", rng1 == rng2)


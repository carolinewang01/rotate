import jax
import jumanji
from jumanji import specs
import matplotlib.pyplot as plt


# Define action sampling function
def sample_action(action_spec, key):
    if isinstance(action_spec, specs.MultiDiscreteArray):
        low = action_spec.minimum
        high = action_spec.maximum
        action = jax.random.randint(key, shape=action_spec.shape, minval=low, maxval=high)
    else:
        action = action_spec.generate_value()
    return action

# Instantiate a Jumanji environment using the registry
env = jumanji.make('LevelBasedForaging-v0')
import pdb; pdb.set_trace()
print("Env name is ", env.name)

NUM_EPISODES = 2

reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)
states = []
key = jax.random.PRNGKey(20394)

for episode in range(NUM_EPISODES):
    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)
    states.append(state)
    while not timestep.last():
        key, action_key = jax.random.split(key)
        observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
        action = sample_action(env.action_spec, action_key)
        print('action is ', action)
        state, timestep = step_fn(state, 
                                  action
                                  )
        print('state is ', state)
        print('timestep is ', timestep) # reward is stored within Timestep object
        # rewards of each agent is given
        
        # env.render(state)
        states.append(state)
    # Freeze the terminal frame to pause the GIF.
    for _ in range(3):
        states.append(state)
    
# anim = env.animate(states, interval=150)
# anim.save("figures/lbf.gif", writer="imagemagick")

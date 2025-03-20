from functools import partial
from jaxmarl.environments import overcooked
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
import jax
import jax.numpy as jnp
from typing import Dict

class OvercookedWrapper(overcooked.Overcooked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def observation_space(self, agent: str):
        return super().observation_space()

    def action_space(self, agent: str):
        return super().action_space()
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = len(self.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OvercookedState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.time

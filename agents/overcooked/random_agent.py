from functools import partial
import jax
import jax.numpy as jnp
# from jaxmarl.environments.overcooked.overcooked import Actions
from typing import Tuple
from flax import struct

@struct.dataclass
class AgentState:
    """Agent state for the random agent."""
    holding: int
    goal: int
    onions_in_pot: int
    rng_key: jax.random.PRNGKey

class RandomAgent:
    """A random agent that takes random actions."""
    
    def __init__(self, agent_name: str):
        self.agent_id = int(agent_name[-1])
        # Initial state - will be passed into and returned from get_action
        self.initial_state = AgentState(
            holding=0,  # Not used but kept for interface consistency
            goal=0,     # Not used but kept for interface consistency
            onions_in_pot=0,  # Not used but kept for interface consistency
            rng_key=jax.random.PRNGKey(self.agent_id)
        )
        
    def get_action(self, obs: jnp.ndarray, state: AgentState = None) -> Tuple[int, AgentState]:
        """Non-jitted version of get_action for initialization purposes"""
        if state is None:
            state = self.initial_state
        return self._get_action(obs, state)
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, state: AgentState) -> Tuple[int, AgentState]:
        """Return a random action and updated state.
        
        Args:
            obs: Flattened observation array (not used)
            state: AgentState containing agent's internal state
            
        Returns:
            Tuple of (random_action, updated_state)
        """
        # Split key for this step
        rng_key, subkey = jax.random.split(state.rng_key)
        
        # Generate random action (excluding Actions.done which is 6)
        action = jax.random.randint(subkey, (), 0, 6)  # Random integer between 0 and 5
        
        # Create new state with updated key
        updated_state = AgentState(
            holding=state.holding,
            goal=state.goal,
            onions_in_pot=state.onions_in_pot,
            rng_key=rng_key
        )
        
        return action, updated_state

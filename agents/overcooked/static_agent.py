from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import Actions
from typing import Tuple
from flax import struct

@struct.dataclass
class AgentState:
    """Agent state for the static agent."""
    holding: int
    goal: int
    onions_in_pot: int
    rng_key: jax.random.PRNGKey

class StaticAgent:
    """A static agent that always takes the stay action."""
    
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
        """Always return the stay action and unchanged state.
        
        Args:
            obs: Flattened observation array (not used)
            state: AgentState containing agent's internal state
            
        Returns:
            Tuple of (stay_action, unchanged_state)
        """
        # Split key for this step (kept for interface consistency)
        rng_key, _ = jax.random.split(state.rng_key)
        return Actions.stay, state

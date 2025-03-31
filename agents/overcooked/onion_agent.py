from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions

from .base_agent import BaseAgent, AgentState, Holding, Goal

class OnionAgent(BaseAgent):
    """A heuristic agent for the Overcooked environment that gets onions 
    and places them in the pot.
    """
    
    def __init__(self, agent_name: str, layout: Dict[str, Any]):
        super().__init__(agent_name, layout)
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, agent_state: AgentState) -> Tuple[int, AgentState]:
        """Get action based on observation and current agentstate.
        
        Args:
            obs: Flattened observation array
            
        Returns:
            Tuple of (action, updated_agent_state)
        """
        # Reshape flattened observation back to 3D
        obs_3d = jnp.reshape(obs, self.obs_shape)
        
        # Define helper functions for each state
        def get_onion(carry):
            '''Go to the nearest onion and pick it up. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "onion", rng_key)
            return (action, new_rng_key)
            
        def put_onion(carry):
            '''Go to the nearest pot and put the onion in it. '''
            obs_3d, rng_key = carry
            action, new_rng_key = self._go_to_obj_and_interact(obs_3d, "pot", rng_key)
            return (action, new_rng_key)
        
        # Get action and update RNG key based on current state
        action, rng_key = lax.cond(
            agent_state.holding == Holding.nothing,
            get_onion,
            lambda carry: lax.cond(
                agent_state.holding == Holding.onion,
                put_onion,
                lambda _: (Actions.stay, carry[1]),
                carry
            ),
            (obs_3d, agent_state.rng_key)
        )
        
        # Create new state with updated key, preserving other state values
        updated_agent_state = AgentState(
            holding=agent_state.holding,
            goal=agent_state.goal,
            onions_in_pot=agent_state.onions_in_pot,
            soup_ready=agent_state.soup_ready,
            rng_key=rng_key
        )

        return action, updated_agent_state
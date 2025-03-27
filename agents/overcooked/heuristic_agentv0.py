from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments.overcooked.overcooked import Actions
from typing import Tuple, Dict, Any
import chex
from flax import struct

# Define agent state constants
HOLDING_NOTHING = 0
HOLDING_ONION = 1
HOLDING_PLATE = 2
HOLDING_DISH = 3  # Completed soup

# Define agent goal states
GOAL_GET_ONION = 0
GOAL_PUT_ONION = 1
GOAL_GET_PLATE = 2
GOAL_GET_SOUP = 3
GOAL_DELIVER = 4

@struct.dataclass
class AgentState:
    """Agent state for the heuristic agent."""
    holding: int
    goal: int
    onions_in_pot: int
    rng_key: jax.random.PRNGKey

class HeuristicAgentV0:
    """A heuristic agent for the Overcooked environment that can create and deliver onion soups."""
    
    def __init__(self, agent_name: str, layout: Dict[str, Any]):
        self.agent_id = int(agent_name[-1])
        self.map_width = layout["width"]
        self.map_height = layout["height"]

        self.num_onion_piles = layout["onion_pile_idx"].shape[0]
        self.num_plate_piles = layout["plate_pile_idx"].shape[0]
        self.num_pots = layout["pot_idx"].shape[0]
        
        self.obs_shape = (self.map_height, self.map_width, 26)  # Overcooked uses 26 channels

        # Initial state - will be passed into and returned from get_action
        self.initial_state = AgentState(
            holding=HOLDING_NOTHING,
            goal=GOAL_GET_ONION,
            onions_in_pot=0,
            rng_key=jax.random.PRNGKey(self.agent_id)
        )
        
    def get_action(self, obs: jnp.ndarray, state: AgentState = None) -> Tuple[int, AgentState]:
        """Non-jitted version of get_action for initialization purposes"""
        if state is None:
            state = self.initial_state
        return self._get_action(obs, state)
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, obs: jnp.ndarray, state: AgentState) -> Tuple[int, AgentState]:
        """Get action and updated state based on observation and current state.
        
        Args:
            obs: Flattened observation array
            state: AgentState containing agent's internal state
            
        Returns:
            Tuple of (action, updated_state)
        """
        # Reshape flattened observation back to 3D
        obs_3d = jnp.reshape(obs, self.obs_shape)
        
        # Get agent's position (channel 0 is a one-hot encoding of the agent's position)
        agent_pos_layer = obs_3d[:, :, 0]
        agent_pos = jnp.argwhere(agent_pos_layer > 0, size=1)[0]
        agent_y, agent_x = agent_pos

        # Get target positions
        pot_y, pot_x = self._get_nearest_pot_pos(obs_3d, agent_y, agent_x)
        onion_x, onion_y = self._get_nearest_onion_or_plate_pos(obs_3d, agent_y, agent_x, "onion")
        plate_x, plate_y = self._get_nearest_onion_or_plate_pos(obs_3d, agent_y, agent_x, "plate")
        
        # select target position 
        target_y, target_x = plate_x, plate_y

        # Move towards target
        nearest_free_y, nearest_free_x = self._get_nearest_free_space(target_y, target_x, obs_3d)
        print(f"Nearest free space: {nearest_free_y}, {nearest_free_x}")

        action = self._move_towards(agent_y, agent_x, 
                nearest_free_y, nearest_free_x, obs_3d, state.rng_key)
        
        # Update state with new RNG key
        new_state = AgentState(
            holding=state.holding,
            goal=state.goal,
            onions_in_pot=state.onions_in_pot,
            rng_key=jax.random.split(state.rng_key)[0]  # Use new key for next step
        )
        
        return action, new_state
    
    def _get_nearest_pot_pos(self, obs: jnp.ndarray, agent_y: int, agent_x: int) -> Tuple[int, int]:
        '''Returns position of the nearest pot.
        
        Args:
            obs: Observation array
            agent_y: Agent's y position
            agent_x: Agent's x position
            
        Returns:
            Tuple of (y, x) coordinates of nearest pot
        '''
        # Get all pot positions
        pot_layer = obs[:, :, 10]
        all_pot_positions = jnp.argwhere(
            pot_layer > 0,
            size=self.num_pots  # Use number of pots from layout
        )

        # Calculate Manhattan distances to each pot
        distances = jnp.sum(
            jnp.abs(all_pot_positions - jnp.array([agent_y, agent_x])),
            axis=1
        )
        
        # Find the position of the nearest pot
        nearest_idx = jnp.argmin(distances)
        nearest_pot_pos = all_pot_positions[nearest_idx]
        
        return nearest_pot_pos

    def _get_nearest_onion_or_plate_pos(self, obs: jnp.ndarray, agent_y: int, agent_x: int,
                                        obj_type: str) -> Tuple[int, int]:
        '''Returns position of the nearest onion or plate.
        Checks both onion piles (channel 12) and onions on counter (channel 23).
        
        Args:
            obs: Observation array
            agent_y: Agent's y position
            agent_x: Agent's x position
            
        Returns:
            Tuple of (y, x) coordinates of nearest onion
        '''
        if obj_type == "onion":
            obj_pile_layer = obs[:, :, 12] # location of onion piles
            obj_layer = obs[:, :, 23] # location of onions on counter
            max_piles = self.num_onion_piles
        elif obj_type == "plate":
            obj_pile_layer = obs[:, :, 14] # location of plate piles
            obj_layer = obs[:, :, 22] # location of plates on counter
            max_piles = self.num_plate_piles
        else:
            raise ValueError(f"Invalid object type: {obj_type}")
        
        # Get location of one pile 
        default_pile_pos = jnp.argwhere(obj_pile_layer > 0, size=1)[0]

        # Combine both layers to get all onion/plate positions
        all_obj_positions = jnp.argwhere(
            jnp.logical_or(obj_pile_layer > 0, obj_layer > 0),
            size=max_piles + 2 # consider at most 2 objects that are not the piles
        )
        # argwhere returns all-zero positions if fewer than max_piles + 2 objects found
        # Replace all-zero positions with default pile position
        is_zero_pos = jnp.all(all_obj_positions == 0, axis=1)
        all_obj_positions = jnp.where(
            jnp.expand_dims(is_zero_pos, axis=1),
            default_pile_pos,
            all_obj_positions
        )
        # Calculate Manhattan distances to each onion/plate
        distances = jnp.sum(
            jnp.abs(all_obj_positions - jnp.array([agent_y, agent_x])),
            axis=1
        )
        
        # Find the position of the nearest onion/plate
        nearest_idx = jnp.argmin(distances)
        nearest_obj_pos = all_obj_positions[nearest_idx]
        
        return nearest_obj_pos
            
    def _get_nearest_free_space(self, y: int, x: int, obs: jnp.ndarray) -> Tuple[int, int]:
        """Get the nearest free space to the target position to figure out where to move.
        Does not account for other agents, which is on purpose.
        
        Args:
            y: Target y position
            x: Target x position
            obs: Observation array
            
        Returns:
            Tuple of (y, x) coordinates of nearest free space
        """
        # Get occupied spaces (walls and counters)
        wall_mask = obs[:, :, 11] > 0       # Channel 11: counter/wall locations
        
        # Check bounds for all four directions
        up_valid = y > 0
        down_valid = y < self.map_height - 1
        right_valid = x < self.map_width - 1
        left_valid = x > 0
        
        # Check if each adjacent position is free
        up_free = up_valid & ~wall_mask[y - 1, x]
        down_free = down_valid & ~wall_mask[y + 1, x]
        right_free = right_valid & ~wall_mask[y, x + 1]
        left_free = left_valid & ~wall_mask[y, x - 1]
        
        # Return first free position found (prioritizing up, down, right, left)
        free_space = lax.cond(
            up_free,
            lambda _: (y - 1, x),
            lambda _: lax.cond(
                down_free,
                lambda _: (y + 1, x),
                lambda _: lax.cond(
                    right_free,
                    lambda _: (y, x + 1),
                    lambda _: lax.cond(
                        left_free,
                        lambda _: (y, x - 1),
                        lambda _: (y, x),  # Fallback to original position if all adjacent spaces are occupied
                        None),
                    None),
                None),
            None)
        return free_space

    @partial(jax.jit, static_argnums=(0,))
    def _move_towards(self, start_y: int, start_x: int, 
                      target_y: int, target_x: int, 
                      obs: jnp.ndarray, key: jax.random.PRNGKey) -> int:
        """Move towards target position while avoiding collisions."""
        # Calculate differences
        x_diff = target_x - start_x
        y_diff = target_y - start_y
        
        # Get occupied spaces (walls, other agent, and counters)
        wall_mask = obs[:, :, 11] > 0       # Channel 11: counter/wall locations
        other_agent_mask = obs[:, :, 1] > 0  # Channel 1: other agent position
        occupied_mask = jnp.logical_or(wall_mask, other_agent_mask)

        # Check if moving in each direction would lead to an occupied space
        up_valid = (start_y > 0) & (~occupied_mask[start_y - 1, start_x])
        down_valid = (start_y < self.map_height - 1) & (~occupied_mask[start_y + 1, start_x])
        right_valid = (start_x < self.map_width - 1) & (~occupied_mask[start_y, start_x + 1])
        left_valid = (start_x > 0) & (~occupied_mask[start_y, start_x - 1])
        stay_valid = True
        
        # Base scores: prefer directions that reduce distance to target (or stay if at target)
        # score encoding: [up, down, right, left, stay]
        up_score, down_score, right_score, left_score, stay_score = 0, 0, 0, 0, 0
        up_score = lax.cond(y_diff > 0, lambda _: 1, lambda _: 0, None)
        down_score = lax.cond(y_diff < 0, lambda _: 1, lambda _: 0, None)        
        right_score = lax.cond(
            y_diff == 0,
            lambda _: lax.cond(x_diff > 0, lambda _: 1, lambda _: 0, None),
            lambda _: 0,
            None)
        
        left_score = lax.cond(
            y_diff == 0,
            lambda _: lax.cond(x_diff < 0, lambda _: 1, lambda _: 0, None),
            lambda _: 0,
            None)
        
        stay_score = lax.cond(
            jnp.logical_and(y_diff == 0, x_diff == 0),
            lambda _: 1,
            lambda _: 0,
            None
        )

        scores = jnp.array([up_score, down_score, right_score, left_score, stay_score])
        
        # Set scores to large negative number for invalid moves
        scores = jnp.where(
            jnp.array([up_valid, down_valid, right_valid, left_valid, stay_valid]),
            scores,
            -1000.0
        )

        print(f"Scores w/o noise: {scores}")

        # Add small random noise to break ties
        key1, key2 = jax.random.split(key)
        noise = jax.random.uniform(key1, shape=(5,), minval=-0.01, maxval=0.01)

        scores += noise
                
        # Choose direction with highest score
        direction = jnp.argmax(scores)
        
        # Map direction to action
        action = lax.switch(
            direction,
            [
                lambda: Actions.up,
                lambda: Actions.down,
                lambda: Actions.right,
                lambda: Actions.left,
                lambda: Actions.stay
            ]
        )
        return action

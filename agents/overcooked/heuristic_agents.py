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

class HeuristicAgent:
    """A heuristic agent for the Overcooked environment that can create and deliver onion soups."""
    
    def __init__(self, agent_name: str, layout: Dict[str, Any]):
        self.agent_id = int(agent_name[-1])
        self.map_width = layout["width"]
        self.map_height = layout["height"]
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
        
        # Extract state values
        holding = state.holding
        goal = state.goal
        onions_in_pot = state.onions_in_pot
        rng_key = state.rng_key
        
        # Split key for this step
        rng_key, subkey = jax.random.split(rng_key)
        
        # Get agent's position (channel 0 for self)
        agent_pos_layer = obs_3d[:, :, 0]
        agent_pos = jnp.argwhere(agent_pos_layer > 0, size=1)[0]
        agent_y, agent_x = agent_pos
        
        # Get agent's orientation (channels 2-5 for self)
        agent_orientation_channels = obs_3d[:, :, 2:6]
        agent_orientation_at_pos = agent_orientation_channels[agent_y, agent_x]
        agent_dir_idx = jnp.argmax(agent_orientation_at_pos)
        
        # Get locations of key objects - ensure we have at least one match 
        pot_locations = jnp.argwhere(obs_3d[:, :, 10] > 0, size=4)
        onion_pile_locations = jnp.argwhere(obs_3d[:, :, 12] > 0, size=2)
        plate_pile_locations = jnp.argwhere(obs_3d[:, :, 14] > 0, size=2)
        delivery_locations = jnp.argwhere(obs_3d[:, :, 15] > 0, size=2)
        
        # Get soup state information
        onions_in_pots = obs_3d[:, :, 16]  # Channel 16: # of onions in pot (0-3)
        pot_cooking_time = obs_3d[:, :, 20]  # Channel 20: cooking time remaining (19-0)
        soup_done = obs_3d[:, :, 21]  # Channel 21: soup done (1 if done)
        
        # Default values
        new_holding = holding
        new_goal = goal
        new_onions_in_pot = onions_in_pot
        
        # Compute nearest pot position and state (if any pots exist)
        pot_exists = pot_locations.shape[0] > 0
        
        # Safe computation of nearest pot that doesn't depend on conditional
        nearest_pot_idx = jnp.argmin(
            jnp.sum(jnp.abs(pot_locations - jnp.array([agent_y, agent_x])), axis=1) + 
            (1000.0 * (1 - jnp.arange(pot_locations.shape[0]) < pot_locations.shape[0])), 
            axis=0
        )
        nearest_pot_pos = jnp.zeros((2,), dtype=jnp.int32).at[0].set(agent_x).at[1].set(agent_y)
        nearest_pot_pos = lax.cond(
            pot_exists, 
            lambda _: pot_locations[nearest_pot_idx],
            lambda _: nearest_pot_pos,
            operand=None
        )
        
        pot_y, pot_x = nearest_pot_pos[0], nearest_pot_pos[1]
        
        # Check if agent is adjacent to nearest pot
        is_adjacent_to_pot = jnp.sum(jnp.abs(nearest_pot_pos - jnp.array([agent_y, agent_x]))) == 1
        
        # Update onions in pot if adjacent
        new_onions_in_pot = lax.cond(
            pot_exists & is_adjacent_to_pot,
            lambda _: jnp.array(onions_in_pots[pot_y, pot_x], dtype=jnp.int32), 
            lambda _: new_onions_in_pot,
            operand=None
        )
        
        # Check pot state
        pot_onions = lax.cond(
            pot_exists,
            lambda _: jnp.array(onions_in_pots[pot_y, pot_x], dtype=jnp.int32),
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=None
        )
        
        pot_cooking = lax.cond(
            pot_exists,
            lambda _: jnp.array(pot_cooking_time[pot_y, pot_x] > 0, dtype=jnp.bool_),
            lambda _: jnp.array(False, dtype=jnp.bool_),
            operand=None
        )
        
        pot_done = lax.cond(
            pot_exists,
            lambda _: jnp.array(soup_done[pot_y, pot_x] > 0, dtype=jnp.bool_),
            lambda _: jnp.array(False, dtype=jnp.bool_),
            operand=None
        )
        
        # Update goals based on pot state and current goal
        pot_full = pot_onions >= 3
        
        # If putting onion and pot is full, switch to getting plate
        new_goal = lax.cond(
            (goal == GOAL_PUT_ONION) & pot_full,
            lambda _: GOAL_GET_PLATE,
            lambda _: new_goal,
            operand=None
        )
        
        # If getting soup and pot is done and holding plate and adjacent to pot, switch to delivery
        new_goal = lax.cond(
            (goal == GOAL_GET_SOUP) & pot_done & (holding == HOLDING_PLATE) & is_adjacent_to_pot,
            lambda _: GOAL_DELIVER,
            lambda _: new_goal,
            operand=None
        )
        
        # Update holding if getting soup and pot is done and holding plate and adjacent to pot
        new_holding = lax.cond(
            (goal == GOAL_GET_SOUP) & pot_done & (holding == HOLDING_PLATE) & is_adjacent_to_pot,
            lambda _: HOLDING_DISH,
            lambda _: new_holding,
            operand=None
        )
        
        # Handle onion pickup
        onion_pile_exists = onion_pile_locations.shape[0] > 0
        
        # Get first onion pile position safely
        onion_pile_pos = lax.cond(
            onion_pile_exists,
            lambda _: onion_pile_locations[0],
            lambda _: jnp.zeros((2,), dtype=jnp.int32),
            operand=None
        )
        
        # Check if adjacent to onion pile
        is_adjacent_to_onion_pile = onion_pile_exists & (jnp.sum(jnp.abs(onion_pile_pos - jnp.array([agent_y, agent_x]))) == 1)
        
        # Update holding and goal if getting onion and adjacent to onion pile
        new_holding = lax.cond(
            (goal == GOAL_GET_ONION) & is_adjacent_to_onion_pile,
            lambda _: HOLDING_ONION,
            lambda _: new_holding,
            operand=None
        )
        
        new_goal = lax.cond(
            (goal == GOAL_GET_ONION) & is_adjacent_to_onion_pile,
            lambda _: GOAL_PUT_ONION,
            lambda _: new_goal,
            operand=None
        )
        
        # Handle putting onion in pot
        is_adjacent_to_pot = pot_exists & (jnp.sum(jnp.abs(nearest_pot_pos - jnp.array([agent_y, agent_x]))) == 1)
        
        # Update holding and goal if putting onion and adjacent to pot and holding onion
        new_holding = lax.cond(
            (goal == GOAL_PUT_ONION) & is_adjacent_to_pot & (holding == HOLDING_ONION),
            lambda _: HOLDING_NOTHING,
            lambda _: new_holding,
            operand=None
        )
        
        # If put onion, either get more onions or get plate depending on pot state
        new_goal = lax.cond(
            (goal == GOAL_PUT_ONION) & is_adjacent_to_pot & (holding == HOLDING_ONION),
            lambda _: lax.cond(new_onions_in_pot >= 3, lambda _: GOAL_GET_PLATE, lambda _: GOAL_GET_ONION, operand=None),
            lambda _: new_goal,
            operand=None
        )
        
        # Handle plate pickup
        plate_pile_exists = plate_pile_locations.shape[0] > 0
        
        # Get first plate pile position safely
        plate_pile_pos = lax.cond(
            plate_pile_exists,
            lambda _: plate_pile_locations[0],
            lambda _: jnp.zeros((2,), dtype=jnp.int32),
            operand=None
        )
        
        # Check if adjacent to plate pile
        is_adjacent_to_plate_pile = plate_pile_exists & (jnp.sum(jnp.abs(plate_pile_pos - jnp.array([agent_y, agent_x]))) == 1)
        
        # Update holding and goal if getting plate and adjacent to plate pile
        new_holding = lax.cond(
            (goal == GOAL_GET_PLATE) & is_adjacent_to_plate_pile,
            lambda _: HOLDING_PLATE,
            lambda _: new_holding,
            operand=None
        )
        
        new_goal = lax.cond(
            (goal == GOAL_GET_PLATE) & is_adjacent_to_plate_pile,
            lambda _: GOAL_GET_SOUP,
            lambda _: new_goal,
            operand=None
        )
        
        # Handle soup pickup
        # Update goal and holding if getting soup and adjacent to done pot and holding plate
        new_holding = lax.cond(
            (goal == GOAL_GET_SOUP) & is_adjacent_to_pot & pot_done & (holding == HOLDING_PLATE),
            lambda _: HOLDING_DISH,
            lambda _: new_holding,
            operand=None
        )
        
        new_goal = lax.cond(
            (goal == GOAL_GET_SOUP) & is_adjacent_to_pot & pot_done & (holding == HOLDING_PLATE),
            lambda _: GOAL_DELIVER,
            lambda _: new_goal,
            operand=None
        )
        
        # Handle delivery
        delivery_loc_exists = delivery_locations.shape[0] > 0
        
        # Get first delivery location safely
        delivery_pos = lax.cond(
            delivery_loc_exists,
            lambda _: delivery_locations[0],
            lambda _: jnp.zeros((2,), dtype=jnp.int32),
            operand=None
        )
        
        # Check if adjacent to delivery location
        is_adjacent_to_delivery = delivery_loc_exists & (jnp.sum(jnp.abs(delivery_pos - jnp.array([agent_y, agent_x]))) == 1)
        
        # Update holding, goal, and onions in pot if delivering and adjacent to delivery and holding dish
        new_holding = lax.cond(
            (goal == GOAL_DELIVER) & is_adjacent_to_delivery & (holding == HOLDING_DISH),
            lambda _: HOLDING_NOTHING,
            lambda _: new_holding,
            operand=None
        )
        
        new_goal = lax.cond(
            (goal == GOAL_DELIVER) & is_adjacent_to_delivery & (holding == HOLDING_DISH),
            lambda _: GOAL_GET_ONION,
            lambda _: new_goal,
            operand=None
        )
        
        new_onions_in_pot = lax.cond(
            (goal == GOAL_DELIVER) & is_adjacent_to_delivery & (holding == HOLDING_DISH),
            lambda _: 0,
            lambda _: new_onions_in_pot,
            operand=None
        )
        
        # Get target location based on current goal
        target_pos = jnp.zeros((2,), dtype=jnp.int32)
        
        # For getting onion, target the onion pile
        target_pos = lax.cond(
            (goal == GOAL_GET_ONION) & onion_pile_exists,
            lambda _: onion_pile_locations[0],
            lambda _: target_pos,
            operand=None
        )
        
        # For putting onion, target the nearest pot
        target_pos = lax.cond(
            (goal == GOAL_PUT_ONION) & pot_exists,
            lambda _: nearest_pot_pos,
            lambda _: target_pos,
            operand=None
        )
        
        # For getting plate, target the plate pile
        target_pos = lax.cond(
            (goal == GOAL_GET_PLATE) & plate_pile_exists,
            lambda _: plate_pile_locations[0],
            lambda _: target_pos,
            operand=None
        )
        
        # For getting soup, first check for done soups, then cooking, then nearest
        has_done_soup = False
        done_soup_pos = jnp.zeros((2,), dtype=jnp.int32)
        
        # This logic is more complex and would need a full rewrite to be jit-compatible
        # For now, we'll just target the nearest pot for getting soup
        target_pos = lax.cond(
            (goal == GOAL_GET_SOUP) & pot_exists,
            lambda _: nearest_pot_pos,
            lambda _: target_pos,
            operand=None
        )
        
        # For delivering, target the delivery location
        target_pos = lax.cond(
            (goal == GOAL_DELIVER) & delivery_loc_exists,
            lambda _: delivery_locations[0],
            lambda _: target_pos,
            operand=None
        )
        
        target_y, target_x = target_pos[0], target_pos[1]
        
        # Decide action - interact if adjacent to target, otherwise move towards it
        is_adjacent_to_target = jnp.sum(jnp.abs(jnp.array([target_y, target_x]) - jnp.array([agent_y, agent_x]))) == 1
        
        action = lax.cond(
            is_adjacent_to_target,
            lambda _: Actions.interact,
            lambda _: self._move_towards(agent_y, agent_x, target_y, target_x, obs_3d, subkey),
            operand=None
        )
        
        # Update state using the dataclass
        updated_state = AgentState(
            holding=new_holding,
            goal=new_goal,
            onions_in_pot=new_onions_in_pot,
            rng_key=rng_key
        )
        
        return action, updated_state
        
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
        
        # Add small random noise to break ties
        key1, key2 = jax.random.split(key)
        noise = jax.random.uniform(key1, shape=(4,), minval=-0.1, maxval=0.1)
        
        # Check if moving in each direction would lead to an occupied space
        up_valid = (start_y > 0) & (~occupied_mask[start_y - 1, start_x])
        down_valid = (start_y < self.map_height - 1) & (~occupied_mask[start_y + 1, start_x])
        right_valid = (start_x < self.map_width - 1) & (~occupied_mask[start_y, start_x + 1])
        left_valid = (start_x > 0) & (~occupied_mask[start_y, start_x - 1])
        
        # Base scores: prefer directions that reduce distance to target
        up_score = -y_diff + noise[0]
        down_score = y_diff + noise[1]
        right_score = x_diff + noise[2] 
        left_score = -x_diff + noise[3]
        
        # Combine scores
        scores = jnp.array([
            up_score * up_valid,
            down_score * down_valid,
            right_score * right_valid,
            left_score * left_valid
        ])
        
        # Set scores to large negative number for invalid moves
        scores = jnp.where(
            jnp.array([up_valid, down_valid, right_valid, left_valid]),
            scores,
            -1000.0
        )
        
        # Choose direction with highest score
        direction = jnp.argmax(scores)
        
        # Map direction to action
        action = lax.switch(
            direction,
            [
                lambda: Actions.up,
                lambda: Actions.down,
                lambda: Actions.right,
                lambda: Actions.left
            ]
        )
        
        # If all moves are invalid, stay put
        all_invalid = jnp.all(~jnp.array([up_valid, down_valid, right_valid, left_valid]))
        
        return lax.cond(
            all_invalid,
            lambda _: Actions.stay,
            lambda _: action,
            None
        )

import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import label

import chex
import jax
from jax import lax
import jax.numpy as jnp

from jaxmarl.environments.overcooked.overcooked import Overcooked, State
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)

from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts
from flax.core.frozen_dict import FrozenDict


class OvercookedV2(Overcooked):
    '''This environment is a modified version of the 
    JaxMARL Overcooked environment that ensures environments are solvable. 
    The main modifications are: 
    - Initialization randomization: Previously, setting `random_reset` would lead to 
        random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, 
        where `random_reset` only controls the initial positions of the two agents. 
    - Initialization of agent positions: Previously, two agents on the same side of a disconnected map. 
        Now, the two agents are always initialized on opposite sides of the map. 
    '''
    
    def __init__(self, 
            layout = FrozenDict(layouts["cramped_room"]),
            random_reset: bool = False,
            random_obj_state: bool = False, 
            max_steps: int = 400,
    ):
        super().__init__(layout=layout, 
                         random_reset=random_reset, 
                         max_steps=max_steps)
        self.random_obj_state = random_obj_state # controls whether pot state and inventory are randomized
    
    def _find_connected_components(self, wall_map: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Find connected components in the map (excluding walls) using JAX operations.
        
        Args:
            wall_map: Boolean mask indicating wall positions (True for walls)
            
        Returns:
            Tuple of (labeled_array, num_features) where:
                - labeled_array: Array where each connected component has a unique label
                - num_features: Number of connected components found
        """
        h, w = wall_map.shape
        # Create a binary mask where 1 represents walkable areas
        walkable = ~wall_map
        
        # Initialize labels array
        labels = jnp.zeros_like(wall_map, dtype=jnp.int32)
        current_label = 1
        
        # Helper function to check if a position is valid and walkable
        def is_valid_pos(pos):
            x, y = pos
            return (x >= 0) & (x < h) & (y >= 0) & (y < w) & walkable[x, y]
        
        # Helper function to get neighbors of a position
        def get_neighbors(pos):
            x, y = pos
            return jnp.array([
                [x-1, y], [x+1, y],  # up, down
                [x, y-1], [x, y+1]   # left, right
            ])
        
        # Flood fill algorithm using JAX operations
        def flood_fill(pos, label):
            x, y = pos
            # If position is already labeled or is a wall, return
            if labels[x, y] != 0:
                return labels
            
            # Label current position
            labels = labels.at[x, y].set(label)
            
            # Get valid neighbors
            neighbors = get_neighbors(pos)
            valid_neighbors = jax.vmap(is_valid_pos)(neighbors)
            valid_neighbors = neighbors[valid_neighbors]
            
            # Recursively label valid neighbors
            for neighbor in valid_neighbors:
                labels = flood_fill(neighbor, label)
            
            return labels
        
        # Find first unlabeled walkable position
        def find_next_unlabeled(pos):
            x, y = pos
            if x >= h:
                return None
            if y >= w:
                return find_next_unlabeled(jnp.array([x + 1, 0]))
            if walkable[x, y] & (labels[x, y] == 0):
                return jnp.array([x, y])
            return find_next_unlabeled(jnp.array([x, y + 1]))
        
        # Main loop to find and label all components
        def label_components(init_state):
            labels, current_label = init_state
            next_pos = find_next_unlabeled(jnp.array([0, 0]))
            
            def cond_fn(state):
                labels, current_label, next_pos = state
                return next_pos is not None
            
            def body_fn(state):
                labels, current_label, next_pos = state
                labels = flood_fill(next_pos, current_label)
                return labels, current_label + 1, find_next_unlabeled(next_pos)
            
            labels, current_label, _ = lax.while_loop(cond_fn, body_fn, (labels, current_label, next_pos))
            return labels, current_label - 1
        
        # Initialize and run the labeling
        labels, num_features = label_components((labels, current_label))
        return labels, num_features

    def _get_component_positions(self, labeled_array, component_idx):
        """Get all positions belonging to a specific component"""
        return jnp.where(labeled_array == component_idx)[0]

    def _initialize_agent_positions(self, key: chex.PRNGKey, wall_map, layout, num_agents: int, all_pos: jnp.ndarray) -> Tuple[chex.PRNGKey, jnp.ndarray]:
        """Initialize agent positions ensuring they are on separate halves of the map if possible.
        
        Args:
            key: JAX PRNG key
            wall_map: Boolean mask indicating wall positions
            layout: Environment layout dictionary
            num_agents: Number of agents to initialize
            all_pos: Array of all possible positions
            
        Returns:
            Tuple of (new_key, agent_idx) where agent_idx contains the initialized agent positions
        """
        # Find connected components in the map
        labeled_array, num_features = self._find_connected_components(wall_map)
        
        if self.random_reset and num_features >= 2:
            # If we have at least 2 components, ensure agents are in different components
            key, subkey = jax.random.split(key)
            # Randomly choose two different components
            component_indices = jax.random.choice(subkey, jnp.arange(1, num_features + 1), shape=(2,), replace=False)
            
            # Get positions for each component
            agent_positions = []
            for i, comp_idx in enumerate(component_indices):
                comp_positions = self._get_component_positions(labeled_array, comp_idx)
                key, subkey = jax.random.split(key)
                # Randomly choose a position from this component
                pos_idx = jax.random.choice(subkey, comp_positions)
                agent_positions.append(pos_idx)
            
            agent_idx = jnp.array(agent_positions)
        else:
            # Use default layout positions or random positions if no components found
            key, subkey = jax.random.split(key)
            agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~wall_map.reshape(-1).astype(jnp.bool_)).astype(jnp.float32), replace=False)
            agent_idx = self.random_reset*agent_idx + (1-self.random_reset)*layout.get("agent_idx", agent_idx)
            
        return key, agent_idx

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state based on `self.random_reset` and `self.random_obj_state`

        If random_reset, agent initial positions are randomized.
        If random_obj_state, pot states and inventory are randomized.

        Environment layout is determined by `self.layout`
        """

        layout = self.layout
        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")
        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Initialize agent positions
        key, agent_idx = self._initialize_agent_positions(key, wall_map, layout, num_agents, all_pos)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose() # dim = n_agents x 2
        occupied_mask = occupied_mask.at[agent_idx].set(1)

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get() # dim = n_agents x 2

        # Keep track of empty counter space (table)
        empty_table_mask = jnp.zeros_like(all_pos)
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24 (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * self.random_obj_state + (1-self.random_obj_state) * jnp.ones((pot_idx.shape[0])) * 23

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                          OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = self.random_obj_state * random_agent_inv + \
                    (1-self.random_obj_state) * jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

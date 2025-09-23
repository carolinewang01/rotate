"""
Two-player iterated simple sabotage game environment following JAXMarl MultiAgentEnv interface.

The payoff matrix is: 
[[1, 0, -1], [0, 1, -1], [-1, -1, -1]]

The game is fully cooperative and both agents have 3 actions: H(ead), T(ail), and S(abotage).
Each agent's observation is the history of joint actions taken by both agents.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import chex
from functools import partial
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces


@chex.dataclass
class SimpleSabotageState:
    """State of the simple sabotage game environment."""
    # Current step in the episode
    step: int
    # History of actions taken by both agents [timestep, agent_id]
    action_history: jnp.ndarray
    # Whether the episode is done
    done: bool


class SimpleSabotage(MultiAgentEnv):
    """
    Two-player iterated simple sabotage game environment.
    
    Actions:
    - 0: Head (H) - Cooperate option 1
    - 1: Tail (T) - Cooperate option 2  
    - 2: Sabotage (S) - Sabotage option
    
    Payoff Matrix (both agents receive same reward):
    [[1, 0, -1], [0, 1, -1], [-1, -1, -1]]
    
    Observations:
    Each agent observes the full history of joint actions taken by both agents.
    """
    
    def __init__(self, max_steps: int = 10, max_history_len: int = 10):
        """
        Initialize the simple sabotage game environment.
        
        Args:
            max_steps: Maximum number of steps in an episode
            max_history_len: Maximum length of action history to track
        """
        super().__init__(num_agents=2)
        self.max_steps = max_steps
        self.max_history_len = max_history_len
        self.agents = ["agent_0", "agent_1"]
        
        # Payoff matrix: [agent0_action, agent1_action] -> reward
        self.payoff_matrix = jnp.array([
            [1, 0, -1],   # agent0 plays H (0)
            [0, 1, -1],   # agent0 plays T (1) 
            [-1, -1, -1]  # agent0 plays S (2)
        ])
        
        # Define action and observation spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: 3 discrete actions for each agent
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.agents
        }
        
        # Observation space: flattened history of joint actions
        # Each timestep contributes 2 values (one for each agent's action)
        # We pad with -1 for unused history slots
        obs_dim = self.max_history_len * 2  # 2 agents
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1, high=2, shape=(obs_dim,), dtype=jnp.int32
            ) for agent in self.agents
        }
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[Dict] = None) -> Tuple[Dict[str, jnp.ndarray], SimpleSabotageState]:
        """
        Reset the environment.
        
        Args:
            key: Random key (unused in this deterministic environment)
            params: Optional parameters (unused)
            
        Returns:
            observations: Dictionary of observations for each agent
            state: Initial environment state
        """
        # Initialize empty action history (padded with -1)
        action_history = jnp.full((self.max_history_len, 2), -1, dtype=jnp.int32)
        
        # Create initial state
        state = SimpleSabotageState(
            step=0,
            action_history=action_history,
            done=False
        )
        
        # Get initial observations (empty history)
        observations = self._get_observations(state)
        
        return observations, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: chex.PRNGKey, 
        state: SimpleSabotageState, 
        actions: Dict[str, int],
        params: Optional[Dict] = None
    ) -> Tuple[Dict[str, jnp.ndarray], SimpleSabotageState, Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Step the environment forward.
        
        Args:
            key: Random key (unused)
            state: Current environment state
            actions: Dictionary of actions for each agent
            params: Optional parameters (unused)
            
        Returns:
            observations: New observations for each agent
            new_state: Updated environment state
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional information
        """
        # Extract actions
        action0 = actions["agent_0"]
        action1 = actions["agent_1"] 
        
        # Calculate rewards using payoff matrix
        reward = self.payoff_matrix[action0, action1]
        rewards = {
            "agent_0": reward,
            "agent_1": reward  # Fully cooperative game
        }
        
        # Update action history
        new_action_history = state.action_history.at[state.step].set(jnp.array([action0, action1]))
        
        # Check if episode is done
        new_step = state.step + 1
        done = new_step >= self.max_steps
        
        # Create new state
        new_state = SimpleSabotageState(
            step=new_step,
            action_history=new_action_history,
            done=done
        )
        
        # Get new observations
        observations = self._get_observations(new_state)
        
        # Create done dictionary
        dones = {
            "agent_0": done,
            "agent_1": done,
            "__all__": done
        }
        
        # Create info dictionary
        # IPPO expects info values to be arrays that can be reshaped
        step_array = jnp.full((self.num_agents,), new_step)
        infos = {
            "step": step_array
        }
        
        return observations, new_state, rewards, dones, infos
    
    def _get_observations(self, state: SimpleSabotageState) -> Dict[str, jnp.ndarray]:
        """
        Get observations for all agents.
        
        Args:
            state: Current environment state
            
        Returns:
            Dictionary of observations for each agent
        """
        # Flatten action history: shape (max_history_len * 2,)
        obs = state.action_history.flatten()
        
        # Both agents see the same observation (full history)
        observations = {
            "agent_0": obs,
            "agent_1": obs
        }
        
        return observations
    
    def observation_space(self, agent: str) -> spaces.Box:
        """Get observation space for an agent."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str) -> spaces.Discrete:
        """Get action space for an agent."""
        return self.action_spaces[agent]
    
    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: SimpleSabotageState) -> Dict[str, jnp.ndarray]:
        """
        Get available actions for each agent.
        All actions are always available.
        """
        avail_actions = jnp.ones(3, dtype=jnp.int32)
        return {
            "agent_0": avail_actions,
            "agent_1": avail_actions
        }
    
    def render(self, state: SimpleSabotageState, mode: str = "human") -> Optional[Any]:
        """
        Render the current state.
        
        Args:
            state: Current environment state
            mode: Rendering mode
            
        Returns:
            Rendered output (implementation depends on mode)
        """
        if mode == "human":
            print(f"Step: {state.step}/{self.max_steps}")
            print("Action History (Agent0, Agent1):")
            
            action_names = ["H", "T", "S"]
            for i in range(min(state.step, self.max_history_len)):
                if state.action_history[i, 0] >= 0:  # Valid entry
                    a0_name = action_names[state.action_history[i, 0]]
                    a1_name = action_names[state.action_history[i, 1]]
                    print(f"  Step {i}: ({a0_name}, {a1_name})")
            
            if state.done:
                print("Episode finished!")
        
        return None
    
    @property 
    def name(self) -> str:
        """Environment name."""
        return "SimpleSabotage"
    
    @property
    def num_actions(self) -> int:
        """Number of actions per agent."""
        return 3
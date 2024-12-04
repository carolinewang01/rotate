import jax
from functools import partial
from typing import Dict, Any
import jax.numpy as jnp
from jumanji.env import Environment as JumanjiEnv
from jumanji import specs as jumanji_specs
from jaxmarl.environments import spaces as jaxmarl_spaces

class JumanjiToJaxMARL(object):
    """Use a Jumanji Environment within JaxMARL.
    Warning: this wrapper has only been tested with LBF.
    """
    def __init__(self, env: JumanjiEnv):
        self.env = env
        self.num_agents = env.num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Adjust action and observation spaces
        # TODO: understand why there are per-agent and non-per agent converter functions
        self.observation_spaces = {
            agent: self._convert_jumanji_spec_to_jaxmarl_space_per_agent(env.observation_spec)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: self._convert_jumanji_action_spec_to_jaxmarl_space(env.action_spec, agent_idx)
            for agent_idx, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        state, timestep = self.env.reset(key)
        obs = self._extract_observations(timestep.observation)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        # Convert dict of actions to array
        actions_array = self._actions_to_array(actions)
        state, timestep = self.env.step(state, actions_array)
        obs = self._extract_observations(timestep.observation)
        reward = self._extract_rewards(timestep.reward)
        done = self._extract_dones(timestep)
        info = timestep.extras
        return obs, state, reward, done, info

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _extract_observations(self, observation):
        # Extract per-agent observations
        obs = {}
        for i in range(self.num_agents):
            agent_obs = {
                "agents_view": observation.agents_view[i],
                "action_mask": observation.action_mask[i],
                "step_count": observation.step_count,
            }
            obs[self.agents[i]] = agent_obs
        return obs

    def _actions_to_array(self, actions: Dict[str, Any]):
        # Convert dict of actions to array
        actions_array = jnp.array([actions[agent] for agent in self.agents], dtype=jnp.int32)
        return actions_array

    def _extract_rewards(self, reward):
        # Extract per-agent rewards
        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        return rewards

    def _extract_dones(self, timestep):
        # Extract per-agent done flags
        done = timestep.last() # jumanji lbf returns a single boolean done for all agents
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        return dones

    def _convert_jumanji_spec_to_jaxmarl_space_per_agent(self, spec: jumanji_specs.Spec):
        """Converts the observation spec for each agent to a JaxMARL space."""
        if hasattr(spec, '__dict__'):
            spaces = {}
            for key, value in spec.__dict__.items():
                if isinstance(value, jumanji_specs.Spec):
                    # Extract per-agent spec
                    if key == 'agents_view':
                        # Each agent's view
                        per_agent_spec = self._get_per_agent_spec(value)
                        spaces['agents_view'] = self._convert_jumanji_spec_to_jaxmarl_space(per_agent_spec)
                    elif key == 'action_mask':
                        per_agent_spec = self._get_per_agent_spec(value)
                        spaces['action_mask'] = self._convert_jumanji_spec_to_jaxmarl_space(per_agent_spec)
                    else:
                        # Handle step_count or other fields
                        spaces[key] = self._convert_jumanji_spec_to_jaxmarl_space(value)
            return jaxmarl_spaces.Dict(spaces)
        else:
            return self._convert_jumanji_spec_to_jaxmarl_space(spec)

    def _get_per_agent_spec(self, spec: jumanji_specs.Spec):
        """Extracts the per-agent spec from a batched spec."""
        if isinstance(spec, jumanji_specs.BoundedArray):
            per_agent_shape = spec.shape[1:]
            # Adjust minimum and maximum if they are arrays matching the shape
            per_agent_min = spec.minimum
            per_agent_max = spec.maximum
            # TODO: check purpose of these lines
            if isinstance(spec.minimum, jnp.ndarray) and spec.minimum.shape == spec.shape:
                per_agent_min = spec.minimum()[1:]
            if isinstance(spec.maximum, jnp.ndarray) and spec.maximum.shape == spec.shape:
                per_agent_max = spec.maximum[1:]
            return type(spec)(
                shape=per_agent_shape,
                dtype=spec.dtype,
                minimum=per_agent_min,
                maximum=per_agent_max,
                name=spec.name
            )
        else:
            raise NotImplementedError(f"Spec type {type(spec)} not supported for per-agent extraction.")

    def _convert_jumanji_action_spec_to_jaxmarl_space(self, spec: jumanji_specs.Spec, agent_idx: int):
        """Converts the action spec for each agent to a JaxMARL space."""
        if isinstance(spec, jumanji_specs.MultiDiscreteArray):
            num_actions = spec.num_values[agent_idx]
            return jaxmarl_spaces.Discrete(num_categories=int(num_actions), dtype=spec.dtype)
        elif isinstance(spec, jumanji_specs.DiscreteArray):
            return jaxmarl_spaces.Discrete(num_categories=spec.num_values, dtype=spec.dtype)
        else:
            raise NotImplementedError(f"Spec type {type(spec)} not supported for action spaces.")

    def _convert_jumanji_spec_to_jaxmarl_space(self, spec: jumanji_specs.Spec):
        """Converts a Jumanji spec to a JaxMARL space."""
        if isinstance(spec, jumanji_specs.DiscreteArray):
            return jaxmarl_spaces.Discrete(num_categories=spec.num_values, dtype=spec.dtype)
        elif isinstance(spec, jumanji_specs.MultiDiscreteArray):
            return jaxmarl_spaces.MultiDiscrete(num_categories=spec.num_values)
        elif isinstance(spec, jumanji_specs.BoundedArray):
            # Handle per-agent minimum and maximum if needed
            return jaxmarl_spaces.Box(
                low=spec.minimum,
                high=spec.maximum,
                shape=spec.shape,
                dtype=spec.dtype
            )
        elif isinstance(spec, jumanji_specs.Array):
            # Assuming unbounded array
            return jaxmarl_spaces.Box(
                low=-jnp.inf,
                high=jnp.inf,
                shape=spec.shape,
                dtype=spec.dtype
            )
        else:
            raise NotImplementedError(f"Spec type {type(spec)} not supported.")

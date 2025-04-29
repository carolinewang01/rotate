from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
import chex

from agents.population_interface import AgentPopulation


class PopulationBuffer(struct.PyTreeNode):
    """PyTree structure to store population parameters with fixed buffer size."""
    params: chex.ArrayTree  # Parameters for all agents in the buffer
    scores: chex.Array  # Score for each agent
    ages: chex.Array  # Age (staleness) of each agent
    filled: chex.Array  # Boolean mask for filled slots
    filled_count: chex.Array  # Count of filled slots
    
    # Configuration parameters
    buffer_size: int = struct.field(pytree_node=False, default=100)
    staleness_coef: float = struct.field(pytree_node=False, default=0.3)
    temp: float = struct.field(pytree_node=False, default=1.0)


class BufferedPopulation(AgentPopulation):
    """Population that maintains a buffer of agent parameters that grows over time.
    
    This extends the AgentPopulation class and provides methods to add new agents
    to the buffer and sample from existing agents based on their scores.
    """
    def __init__(self, max_pop_size, policy_cls, staleness_coef=0.3, temp=1.0):
        """Initialize a buffered population.
        
        Args:
            max_pop_size: Maximum number of agents in the population
            policy_cls: Agent policy class
            staleness_coef: Weight for staleness in sampling
            temp: Temperature for softmax in weighted sampling
        """
        super().__init__(max_pop_size, policy_cls)
        self.max_pop_size = max_pop_size
        self.staleness_coef = staleness_coef
        self.temp = temp
    
    @partial(jax.jit, static_argnums=(0,))
    def reset_buffer(self, example_params):
        """Initialize the buffer with zeros of the appropriate shape.
        
        Args:
            example_params: Example parameters to determine shape and dtype
            
        Returns:
            A new PopulationBuffer
        """
        # Initialize buffer with zeros like example_params
        params = jax.tree_map(
            lambda x: jnp.zeros((self.max_pop_size,) + x.shape, x.dtype),
            example_params
        )
        
        return PopulationBuffer(
            params=params,
            scores=jnp.ones(self.max_pop_size),  # Default score of 1
            ages=jnp.zeros(self.max_pop_size, dtype=jnp.int32),
            filled=jnp.zeros(self.max_pop_size, dtype=bool),
            filled_count=jnp.zeros(1, dtype=jnp.int32),
            buffer_size=self.max_pop_size,
            staleness_coef=self.staleness_coef,
            temp=self.temp
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_sampling_dist(self, buffer):
        """Get distribution for sampling agents based on scores and staleness.
        
        Args:
            buffer: The population buffer
            
        Returns:
            Distribution over agents
        """
        # Score distribution (use scores only for filled slots)
        score_dist = buffer.scores * buffer.filled / buffer.temp
        score_sum = score_dist.sum()
        score_dist = jnp.where(
            score_sum > 0,
            score_dist / score_sum,
            buffer.filled / jnp.maximum(buffer.filled.sum(), 1)  # Uniform over filled slots
        )
        
        # Staleness distribution
        staleness_scores = buffer.ages * buffer.filled
        staleness_sum = staleness_scores.sum()
        staleness_dist = jnp.where(
            staleness_sum > 0,
            staleness_scores / staleness_sum,
            score_dist  # If no staleness, use score distribution
        )
        
        # Combined distribution
        return (1 - buffer.staleness_coef) * score_dist + buffer.staleness_coef * staleness_dist
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_next_insert_idx(self, buffer):
        """Get index for next insertion (either first empty or replace lowest-scored).
        
        Args:
            buffer: The population buffer
            
        Returns:
            Index to insert the next agent
        """
        return jax.lax.cond(
            jnp.less(buffer.filled_count[0], buffer.buffer_size),
            lambda: buffer.filled_count[0],
            lambda: jnp.argmin(self._get_sampling_dist(buffer))
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def add_agent(self, buffer, new_params, score=None):
        """Add a new agent to the buffer.
        
        Args:
            buffer: The population buffer
            new_params: Parameters of the new agent to add
            score: Optional score for the new agent (default to 1.0)
            
        Returns:
            Updated population buffer
        """
        # Find index to insert
        insert_idx = self._get_next_insert_idx(buffer)
        
        # Default score
        if score is None:
            score = 1.0
        
        # Update parameters, filled, count, score, age
        new_buffer_params = jax.tree_map(
            lambda x, p: x.at[insert_idx].set(p),
            buffer.params, new_params
        )
        
        new_filled = buffer.filled.at[insert_idx].set(True)
        new_filled_count = jnp.minimum(buffer.filled_count + 1, jnp.array([buffer.buffer_size]))
        new_scores = buffer.scores.at[insert_idx].set(score)
        
        # Reset age for new agent, increment for others
        ages_plus_one = (buffer.ages + 1) * buffer.filled
        new_ages = ages_plus_one.at[insert_idx].set(0)
        
        return buffer.replace(
            params=new_buffer_params,
            filled=new_filled,
            filled_count=new_filled_count,
            scores=new_scores,
            ages=new_ages
        )
    
    @partial(jax.jit, static_argnums=(0, 2))
    def sample_agent_indices(self, buffer, n, rng, needs_resample_mask=None):
        """Sample n agent indices from the buffer based on scores and staleness.
           Optionally takes a mask to indicate which samples are actually used,
           for correct age updates.

        Args:
            buffer: The population buffer
            n: Number of indices to sample (must be static, typically num_envs)
            rng: Random key
            needs_resample_mask: Optional boolean mask of shape (n,) indicating
                                 which sampled indices are actually used.
                                 If None, assumes all n are used.

        Returns:
            indices: Indices of sampled agents (shape (n,))
            new_buffer: Updated buffer with correctly incremented ages
        """
        rng, sample_rng = jax.random.split(rng, 2)

        buffer_has_agents = jnp.greater(buffer.filled.sum(), 0)
        sampling_dist = self._get_sampling_dist(buffer)

        def handle_empty_buffer():
            # If buffer is empty, return random indices and unchanged buffer
            rand_indices = jax.random.randint(sample_rng, (n,), 0, buffer.buffer_size)
            return rand_indices, buffer

        def handle_filled_buffer():
            # Always sample n indices
            indices = jax.random.choice(
                sample_rng, jnp.arange(buffer.buffer_size),
                shape=(n,), p=sampling_dist, replace=True
            )

            # --- Correct Age Update Logic ---
            # Initialize reset mask for buffer entries
            reset_mask = jnp.zeros(buffer.buffer_size, dtype=bool)

            # Default to assuming all samples are used if mask is not provided
            actual_needs_resample_mask = needs_resample_mask if needs_resample_mask is not None else jnp.ones(n, dtype=bool)

            # Loop through samples to build the reset_mask without dynamic slicing
            def update_reset_mask(i, current_reset_mask):
                buffer_idx = indices[i]
                is_needed = actual_needs_resample_mask[i]
                # Set reset_mask[buffer_idx] = True only if this sample was needed
                new_reset_mask = jax.lax.cond(
                    is_needed,
                    lambda mask: mask.at[buffer_idx].set(True),
                    lambda mask: mask,
                    current_reset_mask
                )
                return new_reset_mask

            reset_mask = jax.lax.fori_loop(0, n, update_reset_mask, reset_mask)

            # Now compute final ages using the statically constructed reset_mask
            increment_mask = buffer.filled & (~reset_mask) # Increment only if filled AND not reset

            # Apply updates
            new_ages = buffer.ages
            new_ages = jnp.where(increment_mask, new_ages + 1, new_ages)
            new_ages = jnp.where(reset_mask, 0, new_ages) # Reset age if it was sampled and needed
            # --- End Age Update Logic ---

            new_buffer = buffer.replace(ages=new_ages)
            return indices, new_buffer

        indices, new_buffer = jax.lax.cond(
            buffer_has_agents,
            handle_filled_buffer,
            handle_empty_buffer
        )

        return indices, new_buffer
    
    def gather_agent_params(self, buffer: PopulationBuffer, agent_indices):
        """Gather the parameters of agents specified by agent_indices.
        
        Args:
            buffer: The population buffer
            agent_indices: Indices with shape (num_envs,), each in [0, buffer_size)
            
        Returns:
            Gathered parameters with shape (num_envs, ...)
        """
        def gather_leaf(leaf):
            # leaf shape: (buffer_size, ...)
            return jax.vmap(lambda idx: leaf[idx])(agent_indices)
        return jax.tree_map(gather_leaf, buffer.params)
    
    def get_actions(self, buffer, agent_indices, obs, done, avail_actions, hstate, rng, 
                    env_state=None, aux_obs=None, test_mode=False):
        """Get actions from agents in the buffer.
        
        Args:
            buffer: The population buffer 
            agent_indices: Indices with shape (num_envs,), each in [0, buffer_size)
            obs: Observations with shape (num_envs, ...)
            done: Done flags with shape (num_envs,)
            avail_actions: Available actions with shape (num_envs, num_actions)
            hstate: Hidden state with shape (num_envs, ...) or None
            rng: Random key
            env_state: Environment state with shape (num_envs, ...) or None
            aux_obs: Optional auxiliary vector to append to observation
            test_mode: Whether to use test mode (deterministic actions)
            
        Returns:
            actions: Actions with shape (num_envs,)
            new_hstate: New hidden state with shape (num_envs, ...) or None
        """
        gathered_params = self.gather_agent_params(buffer, agent_indices)
        num_envs = agent_indices.squeeze().shape[0]
        rngs_batched = jax.random.split(rng, num_envs)
        
        vmapped_get_action = jax.vmap(partial(self.policy_cls.get_action, 
                                             aux_obs=aux_obs, 
                                             env_state=env_state, 
                                             test_mode=test_mode))
        actions, new_hstate = vmapped_get_action(
            gathered_params, obs, done, avail_actions, hstate, 
            rngs_batched)
        return actions, new_hstate 
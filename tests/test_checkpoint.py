from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from common.save_load_utils import load_train_run
# from common.plot_utils import get_stats

@partial(jax.jit, static_argnames=['stats'])
def get_stats(metrics, stats: tuple):
    '''
    Computes mean and std of metrics of interest for each seed and update, 
    using only the final steps of episodes. Note that each rollout contains multiple episodes.

    metrics is a pytree where each leaf has shape 
        (..., rollout_length, num_envs)
    stats is a tuple of strings, each corresponding to a metric of interest in metrics
    '''
    # Get mask for final steps of episodes
    mask = metrics["returned_episode"]
    
    # Initialize output dictionary
    all_stats = {}
    stats = list(stats) # convert to list to correctly iterate if the tuple only has a single element
    for stat_name in stats:
        # Get the metric array
        metric_data = metrics[stat_name]  # Shape: (..., rollout_length, num_envs)

        # Compute means and stds for each seed and update
        # Use masked operations to only consider final episode steps
        means = jnp.where(mask, metric_data, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        # For std, first compute masked values
        masked_vals = jnp.where(mask, metric_data, 0)
        squared_diff = (masked_vals - means[..., None, None]) ** 2
        variance = jnp.where(mask, squared_diff, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        stds = jnp.sqrt(variance)
        # Stack means and stds
        all_stats[stat_name] = jnp.stack([means, stds], axis=-1)
    return all_stats

path = "results/overcooked-v1/cramped_room/fcp/baselines-v0/2025-04-24_13-47-19/ego_train_run"
out = load_train_run(path)
train_metrics = out["metrics"] # leaves have shape (n_ego_seeds, n_updates, rollout_len, n_envs)

train_metrics_exp = jax.tree.map(lambda x: x[np.newaxis, np.newaxis, ...], train_metrics)
stats = get_stats(train_metrics_exp, ('base_return',))
print(stats.keys())
print(stats['base_return'].shape)

# # train_metrics['returned_episode_returns'].shape
# # (1, 1562, 400, 8)
# eval_rets = train_metrics['eval_ep_last_info']['returned_episode_returns'] # shape is (1, 1562, 125, 20, 2)
# print("Agent 0 eval return: ", eval_rets[..., 0].mean())
# print("Whole team eval return: ", eval_rets.mean())
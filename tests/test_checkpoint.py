import numpy as np
from common.save_load_utils import load_train_run
from common.plot_utils import get_stats


path = "results/overcooked-v1/cramped_room/fcp/baselines-v0/2025-04-24_13-47-19/ego_train_run"
out = load_train_run(path)
train_metrics = out["metrics"]
actor_loss = np.asarray(train_metrics["actor_loss"]) # expected shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
print(actor_loss.shape)

stats = get_stats(train_metrics, ('base_return'))
print(stats.keys())
import pdb; pdb.set_trace()

# train_metrics['returned_episode_returns'].shape
# (1, 1562, 400, 8)
eval_rets = train_metrics['eval_ep_last_info']['returned_episode_returns'] # shape is (1, 1562, 125, 20, 2)
print("Agent 0 eval return: ", eval_rets[..., 0].mean())
print("Whole team eval return: ", eval_rets.mean())
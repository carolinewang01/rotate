import jax
import numpy as np
import math
from common.save_load_utils import load_train_run, save_train_run
from common.plot_utils import get_stats


path = "results/lbf/open_ended_minimax/paper-v0:minimax2/2025-05-14_00-42-54/saved_train_run"
outs = load_train_run(path)
teammate_outs, ego_outs = outs

print("Teammate outs: ", teammate_outs.keys()) # dict_keys(['checkpoints', 'final_params', 'metrics])
print("Ego outs: ", ego_outs.keys()) # dict_keys(['checkpoints', 'final_params', 'metrics'])

print("Ego final params shape: ", jax.tree.leaves(ego_outs["final_params"])[0].shape) # (3, 30, 1, 1024)
print("Total parameters: ", sum([math.prod(l.shape) for l in jax.tree.leaves(ego_outs["final_params"])])) # 427507830
# print("Ego checkpoints shape:  ", jax.tree.leaves(ego_outs["checkpoints"])[0].shape) # (3, 30, 5, 1024)

import pdb; pdb.set_trace()

# dest_path = "results/test_sizes/"
# save_train_run(teammate_outs["checkpoints"], dest_path, "tm_checkpoints")
# save_train_run(teammate_outs["final_params"], dest_path, "tm_final_params")
# save_train_run(teammate_outs["metrics"], dest_path, "tm_metrics")
# save_train_run(ego_outs["checkpoints"], dest_path, "ego_checkpoints")
# save_train_run(ego_outs["final_params"], dest_path, "ego_final_params")
# save_train_run(ego_outs["metrics"], dest_path, "ego_metrics")

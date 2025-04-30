import jax
import numpy as np
from common.save_load_utils import load_train_run
from common.plot_utils import get_stats


path = "results/overcooked-v1/cramped_room/oe_persistent_paired/method-explore:uniform/2025-04-30_00-47-07/saved_train_run"
outs = load_train_run(path)
teammate_outs, ego_outs = outs

print("Teammate outs: ", teammate_outs.keys()) # dict_keys(['checkpoints_br', 'checkpoints_conf', 'final_params_br', 'final_params_conf', 'metrics'])
print("Ego outs: ", ego_outs.keys()) # dict_keys(['checkpoints', 'final_buffer', 'final_params', 'metrics'])

final_buffer = ego_outs["final_buffer"] # dict_keys(['ages', 'filled', 'filled_count', 'params', 'scores'])

print("Filled count shape: ", final_buffer["filled_count"].shape) # (1, 30, 1, 1) num_seeds, num_oel_iter, 1, 1
print("Filled shape: ", final_buffer["filled"].shape) # (1, 30, 1, 150) num_seeds, num_oel_iter, 1, max_buffer_size

print("Param leaf shape: ", jax.tree.leaves(final_buffer["params"])[0].shape) # (1, 30, 1, 150, 64) 


import pdb; pdb.set_trace()

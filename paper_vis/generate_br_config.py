'''Script to generate a best response config for the XP matrix, based on the output of the compute_best_response.py script.'''
import json
import os
import yaml
from common.plot_utils import get_metric_names
from paper_vis.plot_globals import TASK_TO_ENV_NAME, get_heldout_agents


OE_DEFAULT_CONFIG = {"actor_type": "s5", "ckpt_key": "final_params", "custom_loader": {"name": "open_ended", "type": "ego"}}
METHOD_TO_CONFIG_VALUES = {
    "ppo_ego_s5": {"actor_type": "s5", "ckpt_key": "final_params"},
    "fcp": {"actor_type": "mlp", "ckpt_key": "final_params", "S5_D_MODEL": 16, "S5_SSM_SIZE": 16, "S5_ACTOR_CRITIC_HIDDEN_DIM": 64, "FC_N_LAYERS": 2},
    "oe_persistent": OE_DEFAULT_CONFIG,
    "oe_paired_resets": OE_DEFAULT_CONFIG,
    "open_ended_minimax": OE_DEFAULT_CONFIG,
    "open_ended_fcp": {**OE_DEFAULT_CONFIG, "S5_D_MODEL": 16, "S5_SSM_SIZE": 16, "S5_ACTOR_CRITIC_HIDDEN_DIM": 64, "FC_N_LAYERS": 2},
}    

METHOD_TO_TRAIN_RUN_NAME = {
    "ppo_ego_s5": "ego_train_run",
    "fcp": "ego_train_run",
    "oe_persistent": "saved_train_run",
    "oe_paired_resets": "saved_train_run",
    "open_ended_minimax": "saved_train_run",
    "open_ended_fcp": "saved_train_run",
}


def generate_br_config(br_json_path: str, task_name: str):
    '''Generate a best response config for a given task, using the output from the compute_best_response.py script, 
    for the heldout agents.
    '''
    with open(br_json_path, 'r') as f:
        computed_br_data = json.load(f)

    # Load metric names for a given environment
    env_name = TASK_TO_ENV_NAME[task_name]
    metric_names = get_metric_names(env_name)

    # load heldout agents
    task_config_path = f"open_ended_training/configs/task/{task_name.replace('-v1', '')}.yaml"
    heldout_agents = get_heldout_agents(task_name, task_config_path)
    heldout_agent_names = list(heldout_agents.keys())

    for m_name in metric_names:
        assert len(computed_br_data[m_name]) == len(heldout_agent_names), \
            f"{m_name} has length {len(computed_br_data[m_name])} but there are {len(heldout_agent_names)} heldout agents"
        method_path_list = computed_br_data[f"{m_name}_method"] # doesn't matter which one we use
        method_seed_list = computed_br_data[f"{m_name}_seed"] # doesn't matter which one we use
        method_iter_list = computed_br_data[f"{m_name}_iter"] # doesn't matter which one we use
    
    br_config = {}
    for i, heldout_agent_name in enumerate(heldout_agent_names):
        hkey = f"{heldout_agent_name.replace(' ', '')}_br"
        br_config[hkey] = {}
        
        method_path = method_path_list[i] 
        method_name = method_path.split(task_name)[1].lstrip('/').split('/')[0]

        br_config[hkey]['path'] = method_path.replace('heldout_eval_metrics', METHOD_TO_TRAIN_RUN_NAME[method_name])
        br_config[hkey]['test_mode'] = False

        # Parse the performance bounds
        br_config[hkey]['performance_bounds'] = {}
        for metric_name in metric_names:
            br_config[hkey]['performance_bounds'][metric_name] = [[0.0, computed_br_data[metric_name][i]]]

        # Parse the idx_list 
        seed_int = method_seed_list[i]
        iter_int = method_iter_list[i]
        br_config[hkey]['idx_list'] = [[seed_int, iter_int]] if iter_int is not None else [seed_int]

        br_config[hkey] = {**br_config[hkey], **METHOD_TO_CONFIG_VALUES[method_name]}

    return br_config

if __name__ == "__main__":
    TASK_NAMES = [
        'lbf', 
        'overcooked-v1/cramped_room', 
        'overcooked-v1/asymm_advantages', 
        'overcooked-v1/forced_coord', 
        'overcooked-v1/counter_circuit', 
        'overcooked-v1/coord_ring'
    ]

    br_config = {}
    for task_name in TASK_NAMES:
        br_json_path = f"results/{task_name}/best_heldout_returns.json"
        br_json_dir = os.path.dirname(br_json_path)
        task_config = generate_br_config(br_json_path=br_json_path, task_name=task_name)
        br_config[task_name] = task_config

    # Dump to yaml file
    out_path = "results/best_heldout_returns_br_config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(br_config, f)
    print(f"Wrote BR config to {out_path}")


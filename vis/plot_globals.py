import jax
import omegaconf
from evaluation.heldout_evaluator import load_heldout_set
from envs import make_env

TASK_TO_ENV_NAME = {
    "lbf": "lbf",
    "overcooked-v1/cramped_room": "overcooked-v1",
    "overcooked-v1/asymm_advantages": "overcooked-v1",
    "overcooked-v1/forced_coord": "overcooked-v1",
    "overcooked-v1/counter_circuit": "overcooked-v1",
    "overcooked-v1/coord_ring": "overcooked-v1",
}

TASK_TO_PLOT_TITLE = {
    "lbf": "Level-Based Foraging",
    "overcooked-v1/cramped_room": "Cramped Room (Overcooked)",
    "overcooked-v1/asymm_advantages": "Asymmetric Advantages (Overcooked)",
    "overcooked-v1/forced_coord": "Forced Coordination (Overcooked)",
    "overcooked-v1/counter_circuit": "Counter Circuit (Overcooked)",
    "overcooked-v1/coord_ring": "Coordination Ring (Overcooked)",
}

TASK_TO_AXIS_DISPLAY_NAME = {
    "lbf": "LBF",
    "overcooked-v1/cramped_room": "CR",
    "overcooked-v1/asymm_advantages": "AA",
    "overcooked-v1/forced_coord": "FC",
    "overcooked-v1/counter_circuit": "CC",
    "overcooked-v1/coord_ring": "CoR",
}

TASK_TO_METRIC_NAME = {
    "lbf": "percent_eaten",
    "overcooked-v1/cramped_room": "base_return",
    "overcooked-v1/asymm_advantages": "base_return",
    "overcooked-v1/forced_coord": "base_return",
    "overcooked-v1/counter_circuit": "base_return",
    "overcooked-v1/coord_ring": "base_return",
}

OE_BASELINES = { # method_path: (type, display_name)
    "open_ended_minimax/paper-v0": ("open_ended", "Minimax"),
    # "open_ended_minimax/paper-v0:minimax2": ("open_ended", "minimax2"), # TODO: remove this result
}

TEAMMATE_GEN_BASELINES = {
    "paired_ued/paper-v0": ("teammate_generation", "PAIRED"),
    "fcp/paper-v0": ("teammate_generation", "FCP"),
    "brdiv/paper-v0": ("teammate_generation", "BRDiv"),
    "comedi/paper-v0": ("teammate_generation", "CoMeDi"),
}

OUR_METHOD = {
    "oe_persistent/paper-v0:1reg": ("open_ended", "ROTATE"),
}

ABLATIONS_OBJ = {
    "oe_persistent/paper-v0:1reg": ("open_ended", "ROTATE (per-state)"),
    "oe_persistent/paper-v0:treg": ("open_ended", "ROTATE (per-traj)"), 
    # "open_ended_paired/paper-v0": ("open_ended", "ROTATE (per-traj, no SXP)"), # TODO: remove locally and rerun this result!
}

ABLATIONS_POP = {
    "ppo_ego_s5/paper-v0:1reg-ego-v-pop": ("teammate_generation", "PPO on ROTATE pop"),
    "oe_paired_resets/paper-v0:1reg": ("open_ended", "ROTATE w/o population"),
}

ROTATE_VARS = {
    "oe_persistent/paper-v0:treg": ("open_ended", "ROTATE (per-traj)"), 
    "oe_persistent/paper-v0:comedi+pop": ("open_ended", "ROTATE+CoMeDi MP"),
    "oe_persistent/paper-v0:paired-treg+pop": ("open_ended", "ROTATE (GAE regret)"),
    "oe_paired_resets/paper-v0:1reg": ("open_ended", "ROTATE w/o population"),
}

SUPPLEMENTAL = {
    # "oe_persistent/paper-v0:comedi+pop": ("open_ended", "ROTATE+CoMeDi MP"),
    # "oe_persistent/paper-v0:paired-treg+pop": ("open_ended", "ROTATE (GAE regret)"),
    # # "oe_persistent/paper-v0:1reg:s5-s": ("open_ended", "rotate, s5-small"),
    # # "oe_persistent/paper-v0:breg:s5-s": ("open_ended", "rotate (obj 0), s5-small"),
    # # "ppo_ego_s5/paper-v0:breg-ego-v-pop": ("teammate_generation", "ppo on rotate pop (obj 0)"),
    "oe_persistent/paper-v0:breg": ("open_ended", "rotate (obj 0) w/sp regret"),
}

RESULTS_DIR = "results_neurips"

GLOBAL_HELDOUT_CONFIG = omegaconf.OmegaConf.load("evaluation/configs/global_heldout_settings.yaml")
CACHE_FILENAME = "cached_summary_metrics.pkl"
HELDOUT_CURVES_CACHE_FILENAME = "cached_heldout_curves.pkl"
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 14

def get_heldout_agents(task_name, task_config_path):
    rng = jax.random.PRNGKey(0)
    heldout_cfg = GLOBAL_HELDOUT_CONFIG["heldout_set"][task_name]
    env_config = omegaconf.OmegaConf.load(task_config_path)
    env_name = env_config["ENV_NAME"]
    env_kwargs = env_config["ENV_KWARGS"]

    env = make_env(env_name, env_kwargs)
    heldout_agents = load_heldout_set(heldout_cfg, env, task_name, env_kwargs, rng)

    return heldout_agents

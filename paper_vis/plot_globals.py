import omegaconf

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

TASK_TO_METRIC_NAME = {
    "lbf": "percent_eaten",
    "overcooked-v1/cramped_room": "base_return",
    "overcooked-v1/asymm_advantages": "base_return",
    "overcooked-v1/forced_coord": "base_return",
    "overcooked-v1/counter_circuit": "base_return",
    "overcooked-v1/coord_ring": "base_return",
}

BASELINES = { # method_path: (type, display_name)
    "open_ended_minimax/paper-v0": ("open_ended", "minimax"),
    "open_ended_paired/paper-v0": ("open_ended", "paired"),
    "fcp/paper-v0": ("teammate_generation", "fcp"),
    "brdiv/paper-v0": ("teammate_generation", "brdiv"),
}

OUR_METHOD = {
    "oe_persistent/paper-v0:breg": ("open_ended", "ours"),
}

ABLATIONS = {
    "oe_persistent/paper-v0:comedi-pop": ("open_ended", "ours w/mixed play"),
    "oe_persistent/paper-v0:paired-treg+pop": ("open_ended", "ours w/traj regret"),
    "oe_paired_resets/paper-v0:breg": ("open_ended", "ours w/o population"),
    "oe_persistent/paper-v0:1reg": ("open_ended", "ours w/o sp-regret"),
}

GLOBAL_HELDOUT_CONFIG = omegaconf.OmegaConf.load("evaluation/configs/global_heldout_settings.yaml")
CACHE_FILENAME = "cached_summary_metrics.pkl"

TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14

import hydra
from omegaconf import OmegaConf

from open_ended_minimax import run_minimax
from open_ended_paired import run_paired
from open_ended_fcp import run_fcp
from open_ended_lagrange import run_lagrange
from paired_ued import run_paired_ued
from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger

@hydra.main(version_base=None, config_path="configs", config_name="base_config_oel")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    if cfg.algorithm["ALG"] == "open_ended_minimax":
        ego_policy, final_ego_params, init_ego_params = run_minimax(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "open_ended_paired":
        ego_policy, final_ego_params, init_ego_params = run_paired(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "paired_ued":
        ego_policy, final_ego_params, init_ego_params = run_paired_ued(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "open_ended_fcp":
        ego_policy, final_ego_params, init_ego_params = run_fcp(cfg, wandb_logger)
    elif cfg.algorithm["ALG"] == "open_ended_lagrange":
        ego_policy, final_ego_params, init_ego_params = run_lagrange(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")

    if cfg["run_heldout_eval"]:
        metric_names = get_metric_names(cfg["task"]["ENV_NAME"])
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(cfg, ego_policy, final_ego_params, init_ego_params)
        log_heldout_metrics(cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names, log_dim0_as_curve=False)

    wandb_logger.close()

if __name__ == '__main__':
    run_training()
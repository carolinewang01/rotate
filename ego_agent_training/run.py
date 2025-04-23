import hydra
from omegaconf import OmegaConf

from ppo_ego import run_ego_training
from common.wandb_visualizations import Logger

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    '''Runs the ego agent training against a fixed partner population. This script is 
    mostly just used for debugging.'''
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)

    run_ego_training(cfg, wandb_logger)
    # Cleanup
    wandb_logger.close()


if __name__ == '__main__':
    run_training()
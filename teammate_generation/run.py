import hydra
from omegaconf import OmegaConf
from common.wandb_visualizations import Logger

from BRDiv import run_brdiv
from train_ego import train_ego_agent

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg["algorithm"]["ALG"] == "brdiv":
        partner_params, partner_population = run_brdiv(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")
    
    if cfg["train_ego"]:
        train_ego_agent(cfg["ego_train_algorithm"], wandb_logger, partner_params, partner_population)
    
    if cfg["heldout_eval"]:
        # TODO: run heldout evaluation
        pass
    
    wandb_logger.close()

if __name__ == '__main__':
    run_training()
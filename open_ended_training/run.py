import hydra
from omegaconf import OmegaConf

from open_ended_minimax import run_minimax
from open_ended_paired import run_paired
from open_ended_fcp import run_fcp

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.algorithm["ALG"] == "minimax":
        run_minimax(cfg)
    elif cfg.algorithm["ALG"] == "paired":
        run_paired(cfg)
    elif cfg.algorithm["ALG"] == "fcp":
        run_fcp(cfg)
    else:
        raise NotImplementedError("Selected method not implemented.")

if __name__ == '__main__':
    run_training()
import hydra
from omegaconf import OmegaConf

from BRDiv import run_brdiv
from L_BRDiv import run_l_brdiv

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.algorithm["ALG"] == "brdiv":
        run_brdiv(cfg)
    elif cfg.algorithm["ALG"] == "lbrdiv":
        run_l_brdiv(cfg)
    else:
        raise NotImplementedError("Selected method not implemented.")

if __name__ == '__main__':
    run_training()
import hydra
from open_ended_minimax import run
from omegaconf import OmegaConf
@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(OmegaConf.to_yaml(hydra_cfg))
    run(cfg)

if __name__ == '__main__':
    run_training()
import hydra
from open_ended_minimax import run

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    #args = vars(args)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run(cfg)

if __name__ == '__main__':
    run_training()
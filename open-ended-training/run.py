import hydra
from open_ended_paired import run_paired
from open_ended_minimax import run_minimax

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    #args = vars(args)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if cfg.algorithm["ALG"] == "minimax":
        run_minimax(cfg)
    elif cfg.algorithm["ALG"] == "PAIRED":
        run_paired(cfg)
    else:
        raise NotImplementedError("Selected method not implemented.")

if __name__ == '__main__':
    run_training()
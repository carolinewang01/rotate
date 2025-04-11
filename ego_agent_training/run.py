import hydra
from omegaconf import OmegaConf
import jax

from ppo_ego import run_ego_training
from common.save_load_utils import load_checkpoints


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    #  Load partner population
    train_partner_path = "results/lbf/fcp_mlp/2025-04-02_15-11-48/fcp_train"
    train_partner_ckpts = load_checkpoints(train_partner_path)
    # ckpts shuld be a dictionary with the key "params"
    # flatten the partner parameters
    n_seeds, m_ckpts = train_partner_ckpts["params"]["Dense_0"]["bias"].shape[:2]
    train_partner_params = jax.tree.map(lambda x: x.reshape((n_seeds * m_ckpts,) + x.shape[2:]), 
                                        train_partner_ckpts)


    run_ego_training(cfg, train_partner_params, pop_size=n_seeds * m_ckpts)

if __name__ == '__main__':
    run_training()
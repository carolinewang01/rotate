'''
This script implements the full training and evaluation pipeline for Fictitious Co-Play (FCP). 
The steps implemented are: 
1. Generate training and testing teammates using IPPO. 
2. Train FCP agent using training teammates. 
3. Evaluate FCP agent against training and testig teammates.
'''
import hydra
import logging
from omegaconf import OmegaConf

from fcp.fcp_train import train_partners_in_parallel, train_fcp_agent
from fcp.fcp_eval import main as eval_main
from fcp.utils import save_train_run, load_checkpoints
from fcp.vis_utils import get_stats, plot_train_metrics

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="fcp_master")
def fcp_pipeline(config):
    '''Runs the full FCP training and evaluation pipeline.'''
    # initialize config
    print(OmegaConf.to_yaml(config, resolve=True))
    config = OmegaConf.to_container(config, resolve=True)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Train/Load partner policies
    train_partner_path = config["PARTNER_TRAIN_ARGS"]["TRAIN_PATH"]
    if train_partner_path is not None:
        train_partner_ckpts = load_checkpoints(train_partner_path)
        log.info(f"Loaded training partners from {train_partner_path}")
    else:
        log.info("Training the training partners...")
        train_out = train_partners_in_parallel(config["PARTNER_TRAIN_ARGS"], config["PARTNER_TRAIN_ARGS"]["TRAIN_SEED"])
        savepath = save_train_run(train_out, savedir, savename="train_partners")
        train_partner_ckpts = train_out["checkpoints"]
        log.info(f"Saved train partner data to {savepath}")

    eval_partner_path = config["PARTNER_TRAIN_ARGS"]["EVAL_PATH"]
    if eval_partner_path is not None:
        eval_partner_ckpts = load_checkpoints(eval_partner_path)
        log.info(f"Loaded testing partners from {eval_partner_path}")
    else:
        log.info("Training the testing partners...")
        eval_out = train_partners_in_parallel(config["PARTNER_TRAIN_ARGS"], config["PARTNER_TRAIN_ARGS"]["EVAL_SEED"])
        eval_partner_ckpts = eval_out["checkpoints"]
        savepath = save_train_run(eval_out, savedir, savename="eval_partners")
        log.info(f"Saved test partner data to {savepath}")

    # Train FCP agent
    log.info("Training FCP Agent...")
    train_cfg = config["FCP_TRAIN_ARGS"]
    fcp_out = train_fcp_agent(train_cfg, train_partner_ckpts)
    savepath = save_train_run(fcp_out, savedir, savename="fcp_train")
    log.info(f"Saved FCP training data to {savepath}")

    # Visualize training metrics
    metrics = fcp_out["metrics"]
    all_stats = get_stats(metrics, ("percent_eaten", "returned_episode_returns"), train_cfg["NUM_ENVS"])
    plot_train_metrics(all_stats, train_cfg["NUM_SEEDS"], train_cfg["NUM_UPDATES"], train_cfg["NUM_STEPS"], train_cfg["NUM_ENVS"])

    # Perform evaluation
    log.info("Evaluating FCP Agent...")
    eval_main(config["FCP_EVAL_ARGS"], eval_savedir=savedir, 
              fcp_ckpts=fcp_out["checkpoints"], 
              train_partner_ckpts=train_partner_ckpts, 
              eval_partner_ckpts=eval_partner_ckpts, 
              num_episodes=32)

    return fcp_out

if __name__ == "__main__":
    # TODO: support wandb logging
    fcp_pipeline()
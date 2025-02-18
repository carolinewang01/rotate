'''
This script implements the full training and evaluation pipeline for Fictitious Co-Play (FCP). 
The steps implemented are: 
1. Generate training and testing teammates using IPPO. 
2. Train FCP agent using training teammates. 
3. Evaluate FCP agent against training and testig teammates.
'''
import hydra
from omegaconf import OmegaConf

from fcp.fcp_train import train_partners_in_parallel, train_fcp_agent
from fcp.fcp_eval import main as eval_main
from fcp.utils import save_train_run, load_checkpoints
from fcp.vis_utils import get_stats, plot_train_metrics


@hydra.main(version_base=None, config_path="config", config_name="fcp_ippo")
def fcp_pipeline(config):
    '''Runs the full FCP training and evaluation pipeline.'''
    # initialize config
    config = OmegaConf.to_container(config)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Train/Load partner policies
    if config['TRAIN_PARTNER_PATH'] is not None:
        train_partner_ckpts = load_checkpoints(config['TRAIN_PARTNER_PATH'])
        print(f"Loaded training partners from {config['TRAIN_PARTNER_PATH']}")
    else:
        # TODO: enable overriding total train time for training partners. 
        train_out = train_partners_in_parallel(config, config["TRAIN_PARTNER_SEED"])
        savepath = save_train_run(savedir, train_out)
        train_partner_ckpts = train_out["checkpoints"]
        print(f"Saved train partner data to {savepath}")

    if config['EVAL_PARTNER_PATH'] is not None:
        eval_partner_ckpts = load_checkpoints(config['EVAL_PARTNER_PATH'])
        print(f"Loaded testing partners from {config['EVAL_PARTNER_PATH']}")
    else:
        eval_out = train_partners_in_parallel(config, config["EVAL_PARTNER_SEED"])
        eval_partner_ckpts = eval_out["checkpoints"]
        savepath = save_train_run(savedir, eval_out)
        print(f"Saved test partner data to {savepath}")

    # Train FCP agent
    print("-------------------------------------", 
          "\nTraining FCP Agent...")
    fcp_out = train_fcp_agent(config, train_partner_ckpts)
    savepath = save_train_run(savedir, fcp_out)
    print(f"Saved FCP training data to {savepath}")

    # Visualize training metrics
    metrics = fcp_out["metrics"]
    all_stats = get_stats(metrics, ("percent_eaten", "returned_episode_returns"), config["NUM_ENVS"])
    plot_train_metrics(all_stats, config["NUM_SEEDS"], config["NUM_UPDATES"], config["NUM_STEPS"], config["NUM_ENVS"])

    # Perform evaluation
    print("-------------------------------------", 
          "\nEvaluating FCP Agent...")
    eval_main(config, eval_savedir=savedir, 
              fcp_ckpts=fcp_out["checkpoints"], 
              train_partner_ckpts=train_partner_ckpts, 
              eval_partner_ckpts=eval_partner_ckpts, 
              num_episodes=32)

    return fcp_out

if __name__ == "__main__":
    # TODO: support wandb logging
    fcp_pipeline()
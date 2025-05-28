# ROTATE

This is the official repository containing code for the paper, "ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork.".

<!-- If you find the code or paper useful, please cite
```
TODO
``` -->

### Cleanup TODOs - ROTATE Release
- Add Figure 1
- <ARXIV RELEASE>
- Make codebase public
- Update README
    - Add citation for paper

## Installation Guide

Follow instructions at `install_instructions.md` to install the necessary libraries. 

Evaluating trained agents against the heldout evaluation set (referred to as $\Pi^\text{eval}$ in the paper) requires downloading the evaluation agents. 
Reproducing the plots from the paper requires the computed best returns achieved against each evaluation agent, which are stated in the paper appendix. 
Directories containing both data can be obtained by running the provided data download script: 
```python download_eval_data.py```

## Quick Guide to Reproduce ROTATE Results
### Experiments

To run the experiments for the ROTATE paper, use the provided bash scripts at `teammate_generation/experiments.sh` and `open_ended_training/experiments.sh`. 
For teammate generation methods (FCP, BRDiv, CoMeDi), please select the algorithm by modifying `teammate_generation/experiments.sh`. Then, to run the method for all tasks, run `bash teammate_generation/experiments.sh`.
Similarly, for the open-ended methods (ROTATE, PAIRED, Minimax Return), please select the algorithm by modifying `open_ended_training/experiments.sh`. To run the method for all tasks, run `bash open_ended_training/experiments.sh`.

As a warning, ROTATE results take a large amount of disk space (~5gB for a training run with 3 seeds).

### Figures 

Code to reproduce the experimental figures in the ROTATE paper are provided at `paper_vis/`, and 
a general workflow to generate the paper figures is provided at `paper_vis/make_paper_plots.sh`. 
The instructions here assume that you have downloaded the evaluation data already, as specified in the Installation Guide.
 
1. Specify experiment paths at `paper_vis/plot_globals.py`
2. Run `bash paper_vis/make_paper_plots.sh` to generate and save the paper figures. Figures are stored at `results/neurips_figures` by default.

*Note:* The first time that the code is run, it may take a while to to generate the metrics and create the plots---around 5 minutes for each bar chart, and 30 min for each learning curve chart (since all checkpoints' evaluation results must be processed). The first time that a particular experimental result is processed, a cache file is automatically generated and stored within each experimental result directory, which makes subsequent runs of the visualization scripts much faster. 
Cache files can by cleared by running, `python paper_vis/clean_cache_files.py`.

## Code Notes 

### Code Style
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but use agent and population interfaces, along with some common utility functions to avoid code duplication. 

### Code Assumptions
The code makes the following assumptions
- Agent policies are assumed to handle "done" signals and reset internally. 
- Environments are assumed to "auto-reset", i.e. when the episode is done, the step function should check for this and reset the environment if needed.

## Project Guide

The project structure is described here. Additional notes about some folders are provided. 

### Project Structure
- `agents/`: Contains agent related implementations.
- `common/`: Shared utilities and common code.
- `envs/`: Environment implementations and wrappers.
- `evaluation/`: Evaluation and visualization scripts.
- `ego_agent_training/`: all ego agent learning implementations. Currently only supports PPO.
- `marl/`: MARL algorithm implementations. Currently only supports IPPO.
- `open_ended_training`: open-ended learning methods (ROTATE Minimax Return, PAIRED)
- `paper_vis/`: code to generate the plots shown in the paper.
- `teammate_generation/`: teammate generation algorithms (BRDiv, FCP, CoMeDi)
- `tests/`: Test scripts used during development.

### Algorithms Implementations

The algorithms in this codebase are divided into four categories, and each stored in their own directory: 
- MARL algorithms, located at `marl/`
- AHT algorithms
    - Ego agent training methods, located at `ego_agent_training/`
    - Two-stage teammate generation methods, located at `teammate_generation/`
    - Open-ended AHT methods, located at `open_ended_training`

Note that algorithms from the `marl/` and `ego_agent_training/` categories are called as subroutines in the other two categories. 
For example: 
- FCP uses the `marl/ippo` implementation as the teammate generation subroutine
- Two-stage teammate generation methods use `ego_agent_training/ppo_ego.py` as the ego agent training routine. 


#### How to Run an Algorithm on a Task

Within each directory, there is a `run.py` which serves as the entry point for 
all algorithms implemented within the directory. 

We use Hydra to manage algorithm and task configurations. 
In each directory above, there is a `configs/` directory with the 
following subdirectories: 
- `configs/algorithm/`: contains algorithm configs, for each algorithm and task combination
- `configs/hydra/`: contains Hydra settings
- `configs/task/`: contains environment configs necessary to specify a task

Given an algorithm and task, Hydra retrieves the appropriate configs are retrieved from the subdirectories above
and merged into the **master config** at `configs/base_config_<placeholder>.yaml`.
The algorithm and task may be manually specified by modifying the master config, or by using 
Hydra's command line argument support. 

For example, the following command runs Fictitous Co-Play on the Level-Based Foraging (LBF) task: 
```
python teammate_generation/run.py task=lbf algorithm=fcp/lbf
```

#### How Logging Works

By default, results are logged to a local `results/` directory, as specified within the `configs/hydra/hydra_simple.yaml` file for each method type, and to the Weights & Biases (wandb) project specified in the master config.
All metrics are logged using wandb, and are viewable using the wandb web interface. 
Please see the [wandb documentation](https://docs.wandb.ai/) for general information about wandb. 

Logging settings in each master config allowing the user to control whether logging is enabled/disabled. 

### Agents

The `agents` directory contains:
- Heuristic agents for Overcooked and LBF environments
- Various actor critic architectures
- Population and agent interfaces for RL agents

You can test Overcooked heuristic agents by running, `python tests/test_overcooked_agents.py`, 
and the LBF heuristic agents by running, `python tests/test_lbf_agents.py`.

### MARL
The `marl/` directory stores our IPPO implementation. 
To run it with wandb logging and using the configs, run: 
```python marl/run.py task=lbf algorithm=ippo/lbf```
Results are logged via wandb, but can also be viewed locally in the `results` directory.

### Envs
#### Jumanji (LBF)
The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.
`
#### Overcooked-v2
We made some modifications to the JaxMARL Overcooked environment to improve the functionality and ensure environments are solvable.

- Initialization randomization: Previously, setting `random_reset` would lead to random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, where `random_reset` only controls the initial positions of the two agents. 
- Agent initial positions: previously, in a map with disconnected components, it was possible for two agents to be spawned in the same component, making it impossible to solve the task. The Overcooked-v2 environment initializes agents such that one is always spawned on each side of the map.


### Paper Vis

As described in the experiment reproduction section, the code in this directory is used to generate the plots from the ROTATE paper. 
Here, we comment on how the code in this folder computes normalization bounds. 

#### Computing Normalization Bounds
You can compute your own normalization upper bounds using the `paper_vis/compute_best_returns.py` script to walk your `results/` directory to recompute the best seen returns for each evaluation partner. 
For usage, see the bash script, `paper_vis/get_best_returns.sh`.

Alternatively, if you do not wish to recompute the normalization uppper bounds or download the provided normalization bounds, you can use the development performance bounds provided directly in `evaluation/configs/global_heldout_settings.yaml` to normalize the results by setting the `renormalize_metrics` argument of the `load_results_for_task()` function to False.  
Note that the development upper performance bounds are not as high as the normalization upper bounds downloaded by `download_eval_data.py` as they were computed earlier in the project. 

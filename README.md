# ROTATE

This is the official repository containing code for the paper, "ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork.".

<!-- If you find the code or paper useful, please cite
```
TODO
``` -->

### Cleanup TODOs - ROTATE Release
- Add comment to ROTATE code explaining why XSP interaction is there and that it should be disabled
 for maximal efficiency
- Add basic documentation
    - Brief comment on logging system 
    - How to reproduced visualizations in paper
    - Documentation of each folder
- <ARXIV RELEASE>
- Make codebase public
- Update README
    - Add citation for paper

## Installation Guide

Follow instructions at `install_instructions.md`.

Download and unzip evaluation agents from this [link](https://drive.google.com/file/d/1KjBV2GekKdRBiK6QSGe2vYx2ThXlG7X7/view?usp=sharing). The code assumes that the evaluation agents are available in an `eval_teammates/` directory.

## Reproducing Experiments

To run the experiments for the ROTATE paper, use the provided bash scripts at `teammate_generation/experiments.sh` and `open_ended_training/experiments.sh`. 

For teammate generation methods (FCP, BRDiv, CoMeDi), please select the algorithm by modifying `teammate_generation/experiments.sh`. Then, to run the method for all tasks, run `bash teammate_generation/experiments.sh`.
Similarly, for the open-ended methods (ROTATE, PAIRED, Minimax Return), please select the algorithm by modifying `open_ended_training/experiments.sh`. To run the method for all tasks, run `bash open_ended_training/experiments.sh`.


## Code Notes 

### Code Style
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but use agent and population interfaces, along with some common utility functions to avoid code duplication. 

### Code Assumptions
The code makes the following assumptions
- Agent policies are assumed to handle "done" signals and reset internally. 
- Environments are assumed to "auto-reset", i.e. when the episode is done, the step function should check for this and reset the environment if needed.

## Project Guide

The project structure is described here. Brief notes about some folders are provided. 

### Project Structure
- `agents/`: Contains agent related implementations.
- `common/`: Shared utilities and common code.
- `envs/`: Environment implementations and wrappers.
- `evaluation/`: Evaluation and visualization scripts.
- `ego_agent_training/`: all ego agent learning implementations. Currently only supports PPO.
- `marl/`: MARL algorithm implementations. Currently only supports IPPO.
- `open_ended_training`: open-ended learning methods (ROTATE Minimax Return, PAIRED)
- `teammate_generation/`: teammate generation algorithms (BRDiv, FCP, CoMeDi)
- `tests/`: Test scripts used during development.

### Algorithms Implementations

The algorithms in this codebase are divided into four categories, and each stored in their own directory: 
- MARL algorithms, located at `marl/`
- AHT algorithms
    - Ego agent training methods, located at `ego_agent_training/`
    - Two-stage teammate generation methods, located at `teammate_generation/`
    - Open-ended AHT methods, located at `open_ended_training`


TODO: finish the design philosophy section!
<!-- Design philosophy -->
Algorithms from each category reference each other 


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
and merged into the master config at `configs/base_config_<placeholder>.yaml`.
The algorithm and task may be manually specified by modifying the master config, or by using 
Hydra's command line argument support. 

For example, to run Fictitous Co-Play on the Level-Based Foraging (LBF) task, use the following command: 
```
python teammate_generation/run.py task=lbf algorithm=fcp/lbf
```

#### How Logging Works

By default, results are logged to a local `results/` directory, as specified within the `configs/hydra/hydra_simple.yaml` file for each method type, and to the Weights & Biases (wandb) project specified in the master config.
All metrics are logged using wandb, and are viewable using the wandb web interface, so 
we highly recommend using wandb to view results. 
Please see the [wandb documentation](https://docs.wandb.ai/) for setup guides. 

Logging settings allowing the user to control whether logging is enabled/disabled, and whether checkpoints are logged locally, are located within the master config. 

### Agents

The `agents` directory contains:
- Heuristic agents for Overcooked and LBF environments
- Various actor critic architectures
- Population and agent interfaces for RL agents

You can test Overcooked heuristic agents by running, `python tests/test_overcooked_agents.py`, 
and the LBF heuristic agent by running, `python tests/test_lbf_agents.py`.

### MARL
The `marl/` directory stores our IPPO implementation. 
To run it with wandb logging and using the configs, run `python ppo/run_ppo.py`. 
Results are logged via wandb, but can also be viewed locally in the `results` directory.

### Envs
#### Jumanji (LBF)
The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.
`
#### Overcooked-v2
We made some modifications to the JaxMARL Overcooked environment to improve the functionality and ensure environments are solvable.

- Initialization randomization: Previously, setting `random_reset` would lead to random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, where `random_reset` only controls the initial positions of the two agents. 
- Agent initial positions: previously, in a map with disconnected components, it was possible for two agents to be spawned in the same component, making it impossible to solve the task. The Overcooked-v2 environment initializes agents such that one is always spawned on each side of the map.



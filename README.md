# continual-aht

## TODOs

### Evaluation
- Metrics: 
    - BRProx metric    
- Regret-based evaluator: 
    - Figure out how to return regret-maximizing teammates that don't sabotage

### Method Exploration: 
- BufferedPopulation
    - Try using regret-based scores
    - Try decreasing the population size / increasing ego agent learning time

### Baselines 
- FIX LBRDIV!
- Implement PLR style FCP baseline (this requires implementing "growing" the population size between iterations and implementing a curator)
- Implement MEP (we should prioritize MEP over TraGeDi because MEP is stronger)

During rebuttal phase: 
- Implement CoMeDi

### Clean Up Code: 
- Metrics logging - currently, eval_ep_last_info's returned_episode_returns value is visualized, which is problematic because 
 it displays the shaped return in Overcooked. We need to fix this.
- In learners, all calls to run_episodes should specify test mode based on the task-specific config value.
- Open-ended learning implementations
    - Remove conf-br weight argument from paired-style methods
    - Rename reinit-conf, reinit br/ego arguments
- Clean up (L)-BRDiv code
    - Consider merging L-BRDiv and BRDiv implementations
    - Consider making ego and br nets the same 
    - Switch from logging BR/Conf losses to SP/XP losses!
- Clean up IPPO: 
    - Update IPPO to use the general agent interface
    - Update IPPO/FCP implementation to use wandb logging that's more aligned with rest of codebase
- Heuristic agents: 
    - Enable decision-making to account for the available actions
- PPO-ego: 
    - Update this code to resample from the agent population each time the episode is done.
    - Figure out better place to put the initialize_ego_agent helper function
- Hydra configs: 
    - add structured hydra configs for ego agent training, evaluation, and ppo

### Code Assumptions
While cleaning up the code, we should re-examine these assumptions for the benchmark. 
- Agent policies are assumed to handle "done" signals and reset internally. 

## Installation Guide
Follow instructions at `install_instructions.md`

### Project Structure
- `agents/`: Contains heuristic agent implementations
- `common/`: Shared utilities and common code
- `envs/`: Environment implementations and wrappers
- `evaluation/`: Evaluation and visualization scripts
- `examples/`: Example usage scripts
- `ego_agent_training/`: all ego-agent implementations. Currently only PPO
- `teammate_generation/`: teammate generation algorithms
- `open_ended_training`: our open-ended learning methods
- `ppo/`: IPPO algorithm implementation
- `tests/`: Test scripts used during development.

## Project Guide

More details about some folders are provided below. 

### Agents

The `agents` directory contains heuristic for each supported environment. 
Currently, only agents for Overcooked have been implemented.
You can run the Overcooked heuristic agent by running, `python tests/test_overcooked_agents.py`.
Later, we would want to add pretrained agents to this directory as well. 

### Envs
#### Jumanji (LBF)
Currently, the only environment supported from the Jumanji suite is Level-Based Foraging (LBF).

The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.
`
#### Overcooked-v2
We made some modifications to the JaxMARL Overcooked environment to improve the functionality and ensure environments are solvable.

- Initialization randomization: Previously, setting `random_reset` would lead to random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, where `random_reset` only controls the initial positions of the two agents. 
- Agent initial positions: previously, in a map with disconnected components, it was possible for two agents to be spawned in the same component, making it impossible to solve the task. The Overcooked-v2 environment initializes agents such that one is always spawned on each side of the map.


### Fictitious Co-Play (FCP)
The `fcp/` directory stores our Fictitious Co-Play implementation. This implementation was based on JaxMARL's IPPO implementation. 
Our full implementation can be run via `python fcp/run_fcp_pipeline.py`. Results are logged via wandb, but can also be viewed locally in the `results` directory.

The FCP implementation includes several training variants:
- `fcp_train_s5.py`: Training with S5 actor-critic architecture for the ego agent
- `fcp_train_mlp.py`: Training with MLP actor-critic architecture for the ego agent
- `fcp_train_rnn.py`: Training with RNN actor-critic architecture for the ego agent
- `fcp_eval.py`: Evaluation script for FCP agents
- `train_partners.py`: Script for training partner agents using IPPO.

### PPO
The `ppo/` directory stores our IPPO implementation. 
To run it with wandb logging and using the configs, run `python ppo/run_ppo.py`. 
Results are logged via wandb, but can also be viewed locally in the `results` directory.

The `ppo/ippo.py` script can also be ran on its own for debugging purposes.

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but importing a couple common utility functions to avoid code duplication. 

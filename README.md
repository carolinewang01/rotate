# ROTATE

This is the official repository containing code for the paper, "ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork.".
If you find the code or paper useful, please cite
```
TODO
```

### Cleanup TODOs - ROTATE Release
- Get codes to work with RNN ego agent (in upstream caht repo)
- Add comment to ROTATE code explaining why XSP interaction is there and that it should be disabled
 for maximal efficiency
- Add basic documentation
    - Explanation of logging system 
    - How to reproduce experiments in paper
    - How to reproduced visualizations in paper
    - Documentation of each folder
- Update README
    - Add citation for paper

## Installation Guide

Follow instructions at `install_instructions.md`.

Download and unzip evaluation agents from this [link](https://drive.google.com/file/d/1i7vljIsPbImj89Gw5QE15DkRd84I1n11/view?usp=sharing). The code assumes that the evaluation agents are available in an `eval_teammates/` directory.

### Project Structure
- `agents/`: Contains heuristic agent implementations
- `common/`: Shared utilities and common code
- `ego_agent_training/`: PPO ego agent implementation
- `envs/`: Environment wrappers
- `evaluation/`: Evaluation and visualization scripts
- `open_ended_training`: open-ended learning methods (ROTATE, ROTATE variations, Minimax Return, PAIRED)
- `paper_vis`: Visualization scripts for the plots in the paper.
- `ppo/`: IPPO algorithm implementation
- `teammate_generation/`: teammate generation algorithms (BRDiv, FCP, CoMeDi)
- `tests/`: Test scripts used during development.

## Project Guide

More details about some folders are provided below. 

### Agents

The `agents` directory contains heuristic for Overcooked and LBF environments. 
You can run the Overcooked heuristic agent by running, `python tests/test_overcooked_agents.py`.

### PPO
The `ppo/` directory stores our IPPO implementation. 
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

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but use agent and population interfaces, along with some common utility functions to avoid code duplication. 

### Code Assumptions
The code makes the following assumptions
- Agent policies are assumed to handle "done" signals and reset internally. 
- Environments are assumed to "auto-reset", i.e. when the episode is done, the step function should check for this and reset the environment if needed.


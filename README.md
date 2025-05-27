# continual-aht

## TODOs

### Clean Up Code - Benchmark Release
- Move some core PPO code (e.g. loss and GAE computation) into ppo_utils.py
- Hydra configs: 
    - add structured hydra configs for evaluation
- PPO-ego: 
    - Update this code to resample from the agent population each time the episode is done.
- Clean up IPPO: 
    - Update IPPO/FCP implementation to use wandb logging that's more aligned with rest of codebase
- Clean up (L)-BRDiv and CoMeDi code
    - Consider merging L-BRDiv and BRDiv implementations
    - Use run_episodes
    - Consider making ego and br nets the same 
    - Switch from logging BR/Conf losses to SP/XP losses!
- Heuristic agents: 
    - Enable decision-making to account for the available actions
- Move best response computation code to its own directory?
- Metrics logging - currently, eval_ep_last_info's returned_episode_returns value is visualized, which is problematic because it displays the shaped return in Overcooked. We need to fix this.
- Final params - we don't need to store this when the last checkpoint IS the final params, right? Remove the redundancy.

### Evaluation
- Heldout eval teammates: write a script to compute the best response teammates for all heldout agents
- Regret-based evaluator: 
    - Figure out how to return regret-maximizing teammates that don't sabotage

### Baselines 
- FIX LBRDIV! - Arrasy will do this
- Implement PLR style FCP baseline (this requires implementing a regret-based curator)
- Implement MEP (we should prioritize MEP over TraGeDi because MEP is stronger)

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

### PPO
The `ppo/` directory stores our IPPO implementation. 
To run it with wandb logging and using the configs, run `python ppo/run_ppo.py`. 
Results are logged via wandb, but can also be viewed locally in the `results` directory.

The `ppo/ippo.py` script can also be ran on its own for debugging purposes.

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but importing a couple common utility functions to avoid code duplication. 

# continual-aht

### TODOs

Experiment Infrastructure
- Implement wandb logging.
- Move away from pickling Jax pytrees (security + version control concerns.) See https://github.com/alvarobartt/safejax or https://docs.jax.dev/en/latest/_autosummary/jax.export.register_pytree_node_serialization.html

Metrics / Logging: 
- Explore open-ended learning metrics 

Method Exploration: 
- Implement PAIRED
- Implement (L)BR-Div

Environments with Diverse Coordination Conventions: 
- Create LBF task where the objective is for the agent to collect multiple fruits in a certain order. 
    - Hand-code LBF teammate policies that collect the fruits in a certain order. 
- Explore other environments offered by Jumanji and JaxMARL.

### Installation Guide
1. Install [Jumanji](https://github.com/instadeepai/jumanji/tree/main) from source, to get the Level Based Foraging (LBF) environment. 
2. Install the fork of JaxMARL in our codebase via `pip install -e .[algs]` (run from the `jaxmarl_caroline` directory). 
- Note that we have made only minor changes to our JaxMARL fork. It is primarily there to make referencing their baseline scripts easier. 
3. Caroline's conda environment has been dumped to the `environment.yml`, found in the root of this codebase. After installing the above packages from source, please create a conda environment following the `environment.yml`. Don't forget to change the name of the environment and the prefix, within the `.yml` file!  
4. Add the path to this codebase to your PYTHONPATH, either by adding a line to your `.bashrc`, or by setting it as a conda environment variable: `conda env config vars set PYTHONPATH=.:$PYTHONPATH`

### Project Guide
#### Tests
The `tests/` directory stores various test scripts, primarily testing environments and environment wrappers. 

#### Envs
The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.

#### Fictitious Co-Play (FCP)
The `fcp/` directory stores our Fictitious Co-Play implementation. This implementation was based on JaxMARL's IPPO implementation on Overcooked. Note that all FCP scripts train on LBF (using our wrapper) by default. 
- Our implementation can be run via `python fcp/fcp_pipeline.py`. Perform X-forwarding to see plots.

Note that the individual scripts referenced by `fcp/fcp_pipeline.py` can be ran individually, which is useful when debugging.
- To train IPPO agents, run `python fcp/ippo_checkpoints.py`. 
- To train a FCP agent, run `python fcp/fcp_train.py`. 
- To evaluate FCP agents against a pool of evaluation checkpoints, run `python fcp/fcp_eval.py`

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop. 
We follow a similar paradigm, but importing functions to avoid code duplication. 

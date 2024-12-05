# continual-aht

### Project Installation Notes
1. Install [Jumanji](https://github.com/instadeepai/jumanji/tree/main) from source, to get the Level Based Foraging (LBF) environment. 
2. Install the fork of JaxMARL in our codebase via `pip install -e .[algs]` (run from the `jaxmarl_caroline` directory). 
- Note that we have not made any changes to our JaxMARL fork, and it is primarily there to make referencing their baseline scripts easier. 
3. Caroline's conda environment packages have been dumped to the `environment.yml`, found in the root of this codebase. After installing the above packages from source, please create a conda environment following the `environment.yml`. Don't forget to change the name of the environment and the prefix, within the `.yml` file!  
4. Add the path to this codebase to your PYTHONPATH, either by adding a line to your `.bashrc`, or by setting it as a conda environment variable: `conda env config vars set PYTHONPATH=.:$PYTHONPATH`

### Project Guide
The `tests/` directory stores various test scripts, primarily testing environments and environment wrappers. 

The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.

The `lbf_training/` directory stores various training scripts. Note that JaxMARL follows a single-script training paradigm. These training scripts were copied from JaxMARL's baseline examples. 
- To train the JaxMARL IPPO on LBF (using our wrapper), run `python lbf_training/jaxmarl_ippo_rnn_lbf_smax.py`. This script was templated off the JaxMARL Smax example. 
- To train the Jumanji A2C example on LBF (w/o our wrapper), run `python lbf_training/jumanji_a2c_lbf.py`. 

### TODOs
Finishing the wrapper: 
- Add support for LBF environment parameters to the wrapper. 

Creating LBF Sanity Check Experiments: 
- Hand-code LBF teammate policies. 
- Create LBF task where the objective is for the agent to collect the fruits in a certain order. 

Developing better understanding of codebase:
- Following the Overcooked example script, figure out how to vmap the LBF training process across multiple seeds. 
- Figure out why returns logged by the LogWrapper don't seem correct. Do we need an environment specific log wrapper? 

Method Exploration: 
- Check Chris Lu's meta-learning works. Can we find meta-learning implementations in Jax? 
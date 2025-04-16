import copy

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.augmented_layouts import augmented_layouts

def process_generator_args(env_kwargs: dict, default_generator_args: dict):
    '''Helper function to process Jumanji generator args. 
    If env_args and default_generator_args have any key overlap, overwrite 
    args in default_generator_args with those in env_args, deleting those in env_args
    '''
    env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
    generator_args_copy = dict(copy.deepcopy(default_generator_args))
    for key in env_kwargs:
        if key in default_generator_args:
            generator_args_copy[key] = env_kwargs[key]
            del env_kwargs_copy[key]
    return generator_args_copy, env_kwargs_copy

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == "lbf":
        default_generator_args = {"grid_size": 7, "fov": 7, 
                          "num_agents": 2, "num_food": 3, 
                          "max_agent_level": 2, "force_coop": True}
        generator_args, env_kwargs_copy = process_generator_args(env_kwargs, default_generator_args)
        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=LbfGenerator(**generator_args),
                            **env_kwargs_copy)
        env = JumanjiToJaxMARL(env)
        
    elif env_name == 'overcooked-v1':
        default_env_kwargs = {"random_reset": True, "random_obj_state": False, "max_steps": 400}
        env_kwargs_copy = copy.deepcopy(env_kwargs)
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env

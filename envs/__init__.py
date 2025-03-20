import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
import copy
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == 'lbf':
        generator_args = {"grid_size": 8, "fov": 8, 
                          "num_agents": 2, "num_food": 3, 
                          "max_agent_level": 2, "force_coop": False}
        # if env_args and default_generator_args have any key overlap, replace 
        # args in default_generator_args with those in env_args, deleting those in env_args
        env_kwargs_copy = copy.deepcopy(env_kwargs)
        for key in env_kwargs_copy:
            if key in generator_args:
                generator_args[key] = env_kwargs[key]
                del env_kwargs[key]
        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=RandomGenerator(**generator_args),
                            **env_kwargs)
        env = JumanjiToJaxMARL(env)
        
    elif env_name == 'overcooked':
        layout = overcooked_layouts[env_kwargs['layout']]
        env_kwargs["layout"] = layout
        env = jaxmarl.make('overcooked', **env_kwargs)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env

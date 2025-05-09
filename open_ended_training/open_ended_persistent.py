import time
import logging
from functools import partial
import copy

import jax
import jax.numpy as jnp
from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, S5ActorCriticPolicy
from agents.population_buffer import BufferedPopulation, add_partners_to_buffer
from agents.initialize_agents import initialize_s5_agent, initialize_actor_with_double_critic
from common.plot_utils import get_metric_names
from envs import make_env
from envs.log_wrapper import LogWrapper

from open_ended_training.ppo_ego_with_buffer import train_ppo_ego_agent_with_buffer
from open_ended_training.oe_paired_resets import train_regret_maximizing_partners as train_regret_resets_partners, log_metrics as log_regret_resets_metrics
from open_ended_training.oe_paired_comedi import train_regret_maximizing_partners as train_oe_comedi_partners, log_metrics as log_oe_comedi_metrics
# from open_ended_training.open_ended_lagrange import train_lagrange_partners as train_regret_maximizing_partners, log_metrics, linear_schedule_regret
from ppo.ippo import make_train as make_ppo_train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_ippo_partners(config, partner_rng, env):
    '''
    Train a pool IPPO agents w/parameter sharing. 
    Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
    '''
    pretrain_config = config["PRETRAIN_ARGS"]
    config["TOTAL_TIMESTEPS"] = pretrain_config["TOTAL_TIMESTEPS"] // pretrain_config["NUM_AGENTS"]
    config["NUM_CHECKPOINTS"] = pretrain_config["NUM_CHECKPOINTS"]
    config["ACTOR_TYPE"] = "pseudo_actor_with_double_critic"

    rngs = jax.random.split(partner_rng, pretrain_config["NUM_AGENTS"])
    train_jit = jax.jit(jax.vmap(make_ppo_train(config, env)))
    out = train_jit(rngs)
    return out

def linear_schedule_regret(iter_idx, config):
    '''Computes the upper and lower regret thresholds based on the iteration index. 
    Updates the config with the next regret thresholds.'''
    frac = iter_idx / config["NUM_OPEN_ENDED_ITERS"]
    config["LOWER_REGRET_THRESHOLD"] = config["LOWER_REGRET_THRESHOLD_START"] + (config["LOWER_REGRET_THRESHOLD_END"] - config["LOWER_REGRET_THRESHOLD_START"]) * frac
    config["UPPER_REGRET_THRESHOLD"] = config["UPPER_REGRET_THRESHOLD_START"] + (config["UPPER_REGRET_THRESHOLD_END"] - config["UPPER_REGRET_THRESHOLD_START"]) * frac
    return config

def persistent_open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                        partner_population, config, ego_config, env):
    '''
    Train the ego agent against a growing population of regret-maximizing partners.
    Unlike the original implementation, the partner population persists across iterations.
    '''
    prev_ego_params, prev_conf_params, prev_br_params, population_buffer, rng, oel_iter_idx = carry
    rng, partner_rng, ego_rng, conf_init_rng, br_init_rng = jax.random.split(rng, 5)

    config = linear_schedule_regret(oel_iter_idx, config) # update regret thresholds

    # Initialize or reuse confederate parameters based on config
    if config["REINIT_CONF"]:
        init_rngs = jax.random.split(conf_init_rng, config["PARTNER_POP_SIZE"])
        conf_params = jax.vmap(conf_policy.init_params)(init_rngs)
    else:
        conf_params = prev_conf_params

    # Initialize or reuse best response parameters based on config
    if config["REINIT_BR_TO_BR"]:
        init_rngs = jax.random.split(br_init_rng, config["PARTNER_POP_SIZE"])
        br_params = jax.vmap(br_policy.init_params)(init_rngs)
    elif config["REINIT_BR_TO_EGO"]:
        br_params = jax.tree.map(lambda x: x[jnp.newaxis, ...].repeat(config["PARTNER_POP_SIZE"], axis=0), prev_ego_params)
    else:
        br_params = prev_br_params
    
    # Train partner agents with ego_policy
    if config["PARTNER_ALGO"] == "oe_paired_resets":
        train_partners_fn = train_regret_resets_partners
    elif config["PARTNER_ALGO"] == "oe_paired_comedi":
        train_partners_fn = train_oe_comedi_partners
    else:
        raise ValueError(f"Invalid PARTNER_ALGO value: {config['PARTNER_ALGO']}")

    train_out = train_partners_fn(config, env,
                                  ego_params=prev_ego_params, ego_policy=ego_policy,
                                  conf_params=conf_params, conf_policy=conf_policy, 
                                  br_params=br_params, br_policy=br_policy, 
                                  partner_rng=partner_rng)
        
    if config["EGO_TEAMMATE"] == "final":
        all_conf_params = train_out["final_params_conf"]
        
    elif config["EGO_TEAMMATE"] == "all":
        n_ckpts = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]
        all_conf_params = jax.tree.map(
            lambda x: x.reshape((n_ckpts,) + x.shape[2:]), 
            train_out["checkpoints_conf"]
        )
    
    # Add all checkpoints and final parameters of all partners to the buffer
    updated_buffer = add_partners_to_buffer(partner_population, population_buffer, all_conf_params)

    # Train ego agent using the population buffer
    # Sample agents from buffer for training
    ego_out = train_ppo_ego_agent_with_buffer(
        config=ego_config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        population_buffer=updated_buffer  # Pass the buffer to the training function
    )
    
    updated_ego_parameters = ego_out["final_params"]
    updated_conf_parameters = train_out["final_params_conf"]
    updated_br_parameters = train_out["final_params_br"]

    # Remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree_map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_conf_parameters, updated_br_parameters, 
             updated_buffer, rng, oel_iter_idx + 1)
    return carry, (train_out, ego_out)


def train_persistent(rng, env, algorithm_config, ego_config):
    rng, init_ego_rng, init_conf_rng1, init_conf_rng2, init_br_rng, train_rng = jax.random.split(rng, 6)
    
    # Initialize ego agent
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    # Initialize confederate agent
    conf_policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
    )
    init_conf_rngs = jax.random.split(init_conf_rng1, algorithm_config["PARTNER_POP_SIZE"])
    init_conf_params = jax.vmap(conf_policy.init_params)(init_conf_rngs)
    
    assert not (algorithm_config["REINIT_BR_TO_EGO"] and algorithm_config["REINIT_BR_TO_BR"]), "Cannot reinitialize br to both ego and br"
    if algorithm_config["REINIT_BR_TO_EGO"]:
        # initialize br policy to have same architecture as ego policy
        # a bit hacky
        br_policy = S5ActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            d_model=algorithm_config.get("S5_D_MODEL", 16),
            ssm_size=algorithm_config.get("S5_SSM_SIZE", 16),
            n_layers=algorithm_config.get("S5_N_LAYERS", 2),
            blocks=algorithm_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=algorithm_config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
            s5_activation=algorithm_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=algorithm_config.get("S5_DO_NORM", True),
            s5_prenorm=algorithm_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=algorithm_config.get("S5_DO_GTRXL_NORM", True),
        )
    else:
        br_policy = MLPActorCriticPolicy(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
        )
    init_br_rngs = jax.random.split(init_br_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_br_params = jax.vmap(br_policy.init_params)(init_br_rngs)
    
    # Create persistent partner population with BufferedPopulation
    # The max_pop_size should be large enough to hold all agents across all iterations
    # Now we need more space since we're storing all checkpoints

    if algorithm_config["EGO_TEAMMATE"] == "final":
        max_pop_size = algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_OPEN_ENDED_ITERS"]
    elif algorithm_config["EGO_TEAMMATE"] == "all":
        max_pop_size = algorithm_config["PARTNER_POP_SIZE"] * \
                       algorithm_config["NUM_CHECKPOINTS"] * \
                       algorithm_config["NUM_OPEN_ENDED_ITERS"]
    else:
        raise ValueError(f"Invalid EGO_TEAMMATE value: {algorithm_config['EGO_TEAMMATE']}")

    if algorithm_config["PRETRAIN_PPO"]:
        max_pop_size += algorithm_config["PRETRAIN_ARGS"]["NUM_AGENTS"] * algorithm_config["PRETRAIN_ARGS"]["NUM_CHECKPOINTS"]

    # hack to initialize the partner population's conf policy class with the right intializer shape
    conf_policy2, init_conf_params2 = initialize_actor_with_double_critic(algorithm_config, env, init_conf_rng2)
    partner_population = BufferedPopulation(
        max_pop_size=max_pop_size,
        policy_cls=conf_policy2,
        sampling_strategy=algorithm_config["SAMPLING_STRATEGY"],
        staleness_coef=algorithm_config["STALENESS_COEF"],
        temp=algorithm_config["SCORE_TEMP"],
    )
    population_buffer = partner_population.reset_buffer(init_conf_params2)
    
    if algorithm_config["PRETRAIN_PPO"]:
        log.info("Pretraining IPPO partners...")
        pretrain_out = train_ippo_partners(algorithm_config, train_rng, env)
        pretrain_checkpoints = pretrain_out["checkpoints"]
        num_ckpts = algorithm_config["PRETRAIN_ARGS"]["NUM_CHECKPOINTS"] * algorithm_config["PRETRAIN_ARGS"]["NUM_AGENTS"]
        pretrain_checkpoints = jax.tree.map(lambda x: x.reshape((num_ckpts,) + x.shape[2:]), 
                                          pretrain_checkpoints)

        log.info("Done pretraining IPPO partners.")
        population_buffer = add_partners_to_buffer(partner_population, population_buffer, pretrain_checkpoints)

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return persistent_open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                                 partner_population, algorithm_config, ego_config, env)
    
    init_carry = (init_ego_params, init_conf_params, init_br_params, population_buffer, train_rng, 0)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )

    if algorithm_config["PRETRAIN_PPO"]:
        # add pretrain out to the teammate out
        outs[0]["pretrain_out"] = pretrain_out
    return outs

def run_persistent(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # initialize ego config
    ego_config = copy.deepcopy(algorithm_config)
    ego_config["TOTAL_TIMESTEPS"] = algorithm_config["TIMESTEPS_PER_ITER_EGO"]
    EGO_ARGS = algorithm_config.get("EGO_ARGS", {})
    ego_config.update(EGO_ARGS)

    log.info("Starting persistent open-ended PAIRED training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_persistent, 
                env=env, algorithm_config=algorithm_config, ego_config=ego_config
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"Persistent open-ended PAIRED training completed in {end_time - start_time} seconds.")

    # Log metrics (reusing the original PAIRED logging function)
    metric_names = get_metric_names(algorithm_config["ENV_NAME"])

    if algorithm_config["PARTNER_ALGO"] == "oe_paired_resets":
        log_metrics = log_regret_resets_metrics
    elif algorithm_config["PARTNER_ALGO"] == "oe_paired_comedi":
        log_metrics = log_oe_comedi_metrics
    else:
        raise ValueError(f"Invalid PARTNER_ALGO value: {algorithm_config['PARTNER_ALGO']}")
    log_metrics(config, wandb_logger, outs, metric_names)

    # Prepare return values for heldout evaluation
    _, ego_outs = outs
    ego_params = jax.tree_map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)
    ego_policy, init_ego_params = initialize_s5_agent(algorithm_config, env, init_ego_rng)

    return ego_policy, ego_params, init_ego_params
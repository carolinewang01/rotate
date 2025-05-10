import shutil
import time
import logging
from typing import NamedTuple
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import wandb

from envs import make_env
from envs.log_wrapper import LogWrapper
from agents.agent_interface import ActorWithConditionalCriticPolicy
from agents.population_interface import AgentPopulation
from agents.mlp_actor_critic import ActorWithConditionalCritic
from agents.population_buffer import BufferedPopulation, add_partners_to_buffer, get_final_buffer
from common.plot_utils import get_metric_names
from common.ppo_utils import unbatchify
from common.save_load_utils import save_train_run
from agents.initialize_agents import initialize_actor_with_conditional_critic

# Initial PPO training
from ppo.ippo import make_train as make_ppo_train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class XPTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    self_id: jnp.ndarray
    oppo_id: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

def train_comedi_partners(train_rng, env, config):
    num_agents = env.num_agents
    assert num_agents == 2, "This code assumes the environment has exactly 2 agents."

    # Define different minibatch sizes for interactions with ego agent and one with BR agent
    config["NUM_ENVS"] = config["NUM_ENVS_XP"] + config["NUM_ENVS_SP"]
    config["NUM_GAME_AGENTS"] = num_agents
    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]

    # Right now assume control of both agent and its BR
    config["NUM_CONTROLLED_ACTORS"] = config["NUM_ACTORS"]

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (num_agents * config["ROLLOUT_LENGTH"])// config["NUM_ENVS"]
    config["MINIBATCH_SIZE_EGO"] = ((config["NUM_GAME_AGENTS"]-1) * config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]
    config["MINIBATCH_SIZE_BR"] = (config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"]) // config["NUM_MINIBATCHES"]

    def gather_params(partner_params_pytree, idx_vec):
        """
        partner_params_pytree: pytree with all partner params. Each leaf has shape (n_seeds, m_ckpts, ...).
        idx_vec: a vector of indices with shape (num_envs,) each in [0, n_seeds*m_ckpts).

        Return a new pytree where each leaf has shape (num_envs, ...). Each leaf has a sampled
        partner's parameters for each environment.
        """
        # We'll define a function that gathers from each leaf
        # where leaf has shape (n_seeds, m_ckpts, ...), we want [idx_vec[i]] for each i.
        # We'll vmap a slicing function.
        def gather_leaf(leaf):
            def slice_one(idx):
                return leaf[idx]  # shape (...)
            return jax.vmap(slice_one)(idx_vec)

        return jax.tree.map(gather_leaf, partner_params_pytree)

    def make_comedi_agents(config):
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac
        
        def train_ippo_partners(config, partner_rng, env):
            '''
            Train a pool IPPO agents w/parameter sharing. 
            Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
            '''
            pretrain_config = config["PRETRAIN_ARGS"]
            config["TOTAL_TIMESTEPS"] = pretrain_config["TOTAL_TIMESTEPS"] // pretrain_config["NUM_AGENTS"]
            config["NUM_CHECKPOINTS"] = pretrain_config["NUM_CHECKPOINTS"]
            config["ACTOR_TYPE"] = "actor_with_conditional_critic"

            rngs = jax.random.split(partner_rng, pretrain_config["NUM_AGENTS"])
            train_jit = jax.jit(jax.vmap(make_ppo_train(config, env)))
            out = train_jit(rngs)
            return out
        
        def train(rng):

            # Start by training a single PPO agent via self-play
            ppo_rng, init_conf_rng, rng = jax.random.split(rng, 3)
            init_ppo_partner = train_ippo_partners(config, ppo_rng, env)

            # Initialzie a population buffer
            conf_policy2, init_conf_params2 = initialize_actor_with_conditional_critic(config, env, init_conf_rng)
            partner_population = BufferedPopulation(
                max_pop_size=config["PARTNER_POP_SIZE"],
                policy_cls=conf_policy2,
                sampling_strategy=config["SAMPLING_STRATEGY"],
                staleness_coef=config["STALENESS_COEF"],
                temp=config["SCORE_TEMP"],
            )
            population_buffer = partner_population.reset_buffer(init_conf_params2)
            population_buffer = partner_population.add_agent(population_buffer, init_ppo_partner)

            def add_conf_policy(pop_buffer, func_input):
                iter_id, rng = func_input

                # Create new confederate agent policy and critic
                policy, init_params = initialize_actor_with_conditional_critic(
                    config, env, rng
                )

                # Create a train_state and optimizer for the newly initialzied model
                if config["ANNEAL_LR"]:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=linear_schedule, eps=1e-5),
                    )
                else:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
                        optax.adam(config["LR"], eps=1e-5))
                    
                train_state = TrainState.create(
                    apply_fn=policy.network.apply,
                    params=init_params,
                    tx=tx,
                )

                # Reset envs for SP, XP, and MP
                rng, update_rng, reset_rng_eval, reset_rng_sp, reset_rng_xp, reset_rng_mp = jax.random.split(rng, 6)
                
                reset_rngs_evals = jax.random.split(reset_rng_eval, config["NUM_ENVS"])
                reset_rngs_sps = jax.random.split(reset_rng_sp, config["NUM_ENVS"])
                reset_rngs_xps = jax.random.split(reset_rng_xp, config["NUM_ENVS"])
                reset_rngs_mps = jax.random.split(reset_rng_mp, config["NUM_ENVS"])

                obsv_xp_eval, env_state_xp_eval = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_evals)
                obsv_xp, env_state_xp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_sps)
                obsv_sp, env_state_sp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_xps)
                obsv_mp, env_state_mp = jax.vmap(env.reset, in_axes=(0,))(reset_rngs_mps)

                # build a pytree that can hold the parameters for all checkpoints.
                checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
                num_ckpts = config["NUM_CHECKPOINTS"]
                def init_ckpt_array(params_pytree):
                    return jax.tree.map(
                        lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                        params_pytree
                    )

                update_steps = 0
                init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
                update_runner_state = (
                    (
                        train_state, pop_buffer,
                        env_state_xp_eval, obsv_xp_eval, 
                        env_state_sp, obsv_sp, 
                        env_state_xp, obsv_xp,
                        env_state_mp, obsv_mp,
                        init_done, 
                    ), 
                    update_steps
                )

                checkpoint_array = init_ckpt_array(train_state.params)
                ckpt_idx = 0
                update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

                def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
                    update_runner_state, checkpoint_array, ckpt_idx = update_with_ckpt_runner_state
                    (
                        train_state, pop_buffer,
                        env_state_xp_eval, obsv_xp_eval, 
                        env_state_sp, obsv_sp, 
                        env_state_xp, obsv_xp,
                        env_state_mp, obsv_mp,
                        init_done, 
                    ), update_steps = update_runner_state

                    # Step 1
                    # TODO Write a function that outputs XP partner ID from the pop_buffer (Line 5, ALg 2 from Sarkar)
                    # Start rollouts from env_state_xp_eval
                    # In each rollout, trained agent (params in train_state) 
                    # needs to use all the stored confederate policies as its partner
                    # We then average the expected returns when interacting with each conf
                    # policy in the buffer. We use the partner with the highest returns
                    # as the partner for XP.

                    # Step 2
                    # TODO Do self-play (based on train_state params) rollout like in the IPPO code

                    # Step 3 
                    # TODO Do XP rollout (based on train_state params and the param in pop_buffer identified in Step 1)

                    # Step 4
                    # TODO Do MP rollout (based on train_state params and the param in pop_buffer identified in Step 1)

                    # Step 5
                    # TODO Update exactly like in PAIRED CoMeDi

                    return None

                runner_state, metrics = jax.lax.scan(
                    _update_step_with_checkpoint,
                    update_with_ckpt_runner_state,
                    xs=None,  # No per-step input data
                    length=config["NUM_UPDATES"],
                )

                updated_pop_buffer = partner_population.add_agent(pop_buffer, runner_state[0][0].params)
                return updated_pop_buffer
            
            policy_rngs = jax.random.split(config["PARTNER_POP_SIZE"])
            rng, iter_rngs = policy_rngs[0], policy_rngs[1:]
            
            iter_ids = jnp.arange(1, config["PARTNER_POP_SIZE"])
            final_population_buffer = jax.lax.scan(
                add_conf_policy, population_buffer, (iter_ids, iter_rngs)
            )
            
            return final_population_buffer

def get_comedi_population(config, out, env):
    '''
    Get the partner params and partner population for ego training.
    '''
    brdiv_pop_size = config["algorithm"]["PARTNER_POP_SIZE"]

    # partner_params has shape (num_seeds, brdiv_pop_size, ...)
    partner_params = out['final_params_conf']
    
    partner_policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[1]).n,
        obs_dim=env.observation_space(env.agents[1]).shape[0],
        pop_size=brdiv_pop_size, # used to create onehot agent id
        activation=config["algorithm"].get("ACTIVATION", "tanh")
    )

    # Create partner population
    partner_population = AgentPopulation( 
        pop_size=brdiv_pop_size,
        policy_cls=partner_policy
    )

    return partner_params, partner_population

def run_brdiv(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    log.info("Starting CoMeDi training...")
    start = time.time()
    
    # Generate multiple random seeds from the base seed
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    # Create a vmapped version of train_brdiv_partners
    with jax.disable_jit(False):
        vmapped_train_fn = jax.jit(
            jax.vmap(
                partial(train_comedi_partners, env=env, config=algorithm_config)
            )
        )
        out = vmapped_train_fn(rngs)
    
    end = time.time()
    log.info(f"CoMeDi training complete in {end - start} seconds")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)

    partner_params, partner_population = get_comedi_population(config, out, env)

    return partner_params, partner_population


def compute_sp_mask_and_ids(pop_size):
    cross_product = np.meshgrid(
        np.arange(pop_size),
        np.arange(pop_size)
    )
    agent_id_cartesian_product = np.stack([g.ravel() for g in cross_product], axis=-1)
    conf_ids = agent_id_cartesian_product[:, 0]
    ego_ids = agent_id_cartesian_product[:, 1]
    sp_mask = (conf_ids == ego_ids)
    return sp_mask, agent_id_cartesian_product

def log_metrics(config, outs, logger, metric_names: tuple):
    metrics = outs["metrics"]
    # metrics now has shape (num_seeds, num_updates, _, _, pop_size)
    num_seeds, num_updates, _, _, pop_size = metrics["pg_loss_conf_agent"].shape # number of trained pairs

    ### Log evaluation metrics
    # we plot XP return curves separately from SP return curves 
    # shape (num_seeds, num_updates, (pop_size)^2, num_eval_episodes, 1)
    all_returns = np.asarray(metrics["eval_ep_last_info"])
    xs = list(range(num_updates))
    
    sp_mask, agent_id_cartesian_product = compute_sp_mask_and_ids(pop_size)
    sp_returns = all_returns[:, :, sp_mask]
    xp_returns = all_returns[:, :, ~sp_mask]
    
    # Average over seeds, then over agent pairs and episodes
    sp_return_curve = sp_returns.mean(axis=(0, 2, 3, 4))
    xp_return_curve = xp_returns.mean(axis=(0, 2, 3, 4))

    for step in range(num_updates):
        logger.log_item("Eval/AvgSPReturnCurve", sp_return_curve[step], train_step=step)
        logger.log_item("Eval/AvgXPReturnCurve", xp_return_curve[step], train_step=step)
    logger.commit()

    # log final XP matrix to wandb - average over seeds
    last_returns_array = all_returns[:, -1].mean(axis=(0, 2, 3))
    last_returns_array = np.reshape(last_returns_array, (pop_size, pop_size))
    logger.log_xp_matrix("Eval/LastXPMatrix", last_returns_array)

    ### Log population loss as multi-line plots, where each line is a different population member
    # shape (num_seeds, num_updates, update_epochs, num_minibatches, pop_size)
    # Average over seeds
    processed_losses = {
        "ConfPGLoss": np.asarray(metrics["pg_loss_conf_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "BRPGLoss": np.asarray(metrics["pg_loss_br_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "ConfValLoss": np.asarray(metrics["value_loss_conf_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "BRValLoss": np.asarray(metrics["value_loss_br_agent"]).mean(axis=(0, 2, 3)).transpose(),
        "ConfEntropy": np.asarray(metrics["entropy_conf"]).mean(axis=(0, 2, 3)).transpose(),
        "BREntropy": np.asarray(metrics["entropy_br"]).mean(axis=(0, 2, 3)).transpose(),
    }
    
    xs = list(range(num_updates))
    keys = [f"pair {i}" for i in range(pop_size)]
    for loss_name, loss_data in processed_losses.items():
        logger.log_item(f"Losses/{loss_name}", 
            wandb.plot.line_series(xs=xs, ys=loss_data, keys=keys, 
            title=loss_name, xname="train_step")
        )

    ### Log artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Save train run output and log to wandb as artifact
    out_savepath = save_train_run(outs, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    
    # Cleanup locally logged out files
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)

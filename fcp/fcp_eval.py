import jax
import jax.numpy as jnp
import jaxmarl
import jumanji
from jaxmarl.wrappers.baselines import LogWrapper
import pickle

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from fcp.networks import ActorCritic
from fcp.utils import load_checkpoints


def eval_fcp_agent(config, fcp_checkpoints, eval_checkpoints, num_episodes: int):
    '''
    fcp_checkpoints: a pytree where each leaf contains N seeds, and M checkpoints of trained FCP agents 
                    (e.g. {"params": <pytree with leaves of shape (N, M, ...)>})
    eval_checkpoints: a pytree where each leaf contains N seeds, and M checkpoints of trained IPPO agents.
                    All eval checkpoints are treated as one pool of partners.
    For each FCP agent (each seed, each checkpoint) we evaluate it against every eval checkpoint partner in the pool,
    running num_episodes per pairing.
    
    Returns a dictionary with key "episode_returns" whose value is a jnp.array of shape 
       (num_fcp_seeds, num_fcp_ckpts, num_eval_total, num_episodes)
    '''
    # --- 1. Prepare the environment ---
    if config["ENV_NAME"] == 'lbf':
        env = jumanji.make('LevelBasedForaging-v0')
        env = JumanjiToJaxMARL(env)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    # --- 2. Build the networks ---
    fcp_net = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    partner_net = ActorCritic(env.action_space(env.agents[1]).n, activation=config["ACTIVATION"])

    # --- 3. Process checkpoints ---
    # fcp_checkpoints: each leaf shape (n_fcp, m_fcp, ...); extract dimensions.
    fcp_params = fcp_checkpoints["params"]
    sample_leaf = jax.tree_util.tree_leaves(fcp_params)[0]
    n_fcp, m_fcp = sample_leaf.shape[:2]

    # For eval checkpoints, flatten the first two dims to form a partner pool.
    def flatten_ckpt(x):
        return x.reshape(-1, *x.shape[2:])
    eval_params_flat = jax.tree.map(flatten_ckpt, eval_checkpoints["params"])
    sample_leaf_eval = jax.tree_util.tree_leaves(eval_params_flat)[0]
    num_eval_total = sample_leaf_eval.shape[0]

    # We'll use a fixed maximum step count per episode.
    max_episode_steps = 103 # DEBUG MOD

    # --- 4. Inner evaluation functions (without jit) ---
    def run_single_episode(rng, fcp_param, partner_param):
        # Reset the env.
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        # rng and reset_rng.val.val.val.shape are (3, 5, 15, 2)
        # Do one step to get a dummy info structure.
        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
        pi0, _ = fcp_net.apply(fcp_param, obs["agent_0"])
        # note that 3 is num_seeds, 5 is num_checkpoints, 15 is ???, 12 is the shape of the observation
        # obs['agent_0'] shape is (12,), and obs['agent_0].val.val.val.shape is (3, 5, 15, 12)
        # act0 = pi0.sample(seed=act_rng)
        act0 = pi0.mode()
        # act0 shape is (), and act0.val.val.val.shape is (3, 5, 15)
        pi1, _ = partner_net.apply({'params': partner_param}, obs["agent_1"])
        # act1 = pi1.sample(seed=part_rng)
        act1 = pi1.mode()
        both_actions = [act0, act1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        _, _, _, _, dummy_info = env.step(step_rng, env_state, env_act)

        # We'll use a scan to iterate steps until the episode is done.
        ep_ts = 1
        init_carry = (ep_ts, env_state, obs, rng, jnp.array(False), dummy_info)
        def scan_step(carry, _):
            def take_step(carry_step):
                ep_ts, env_state, obs, rng, done_flag, last_info = carry_step
                rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
                pi0, _ = fcp_net.apply(fcp_param, obs["agent_0"])
                # act0 = pi0.sample(seed=act_rng)
                act0 = pi0.mode()
                pi1, _ = partner_net.apply({'params': partner_param}, obs["agent_1"])
                # act1 = pi1.sample(seed=part_rng)
                act1 = pi1.mode()
                both_actions = [act0, act1]
                env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
                obs_next, env_state_next, reward, done, info = env.step(step_rng, env_state, env_act)

                # return (ep_ts + 1, env_state_next, obs_next, rng, done["agent_0"], info)
                return (ep_ts + 1, env_state_next, obs_next, rng, done["__all__"], info)
            
            ep_ts, env_state, obs, rng, done_flag, last_info = carry
            new_carry = jax.lax.cond(
                False, # done_flag,  # DEBUG FLAG
                # if done, execute true function(operand). else, execute false function(operand).
                lambda curr_carry: curr_carry, # True fn
                take_step, # False fn
                operand=carry
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(
            scan_step, init_carry, None, length=max_episode_steps)
        # Return the final info (which includes the episode return via LogWrapper).
        return final_carry[-1]

    def run_episodes(rng, fcp_param, partner_param, num_eps):
        def body_fn(carry, _):
            rng = carry
            rng, ep_rng = jax.random.split(rng)
            ep_info = run_single_episode(ep_rng, fcp_param, partner_param)
            return rng, ep_info
        rng, ep_infos = jax.lax.scan(body_fn, rng, None, length=num_eps)
        return ep_infos  # each leaf has shape (num_eps, ...)

    def eval_pair_fn(rng, fcp_param, partner_param):
        return run_episodes(rng, fcp_param, partner_param, num_episodes)

    # --- 5. Outer function to be jitted and vmapped over fcp seeds ---
    @jax.jit
    def outer_eval(rngs):
        """
        rngs: an array of shape (n_fcp, key), one key per FCP seed.
        Returns a pytree with leaves of shape (n_fcp, m_fcp, num_eval_total, num_episodes, ...).
        """
        # For each seed, use its corresponding fcp parameters.
        # Here, fcp_params has shape (n_fcp, m_fcp, ...).
        def eval_fcp_seed(rng, fcp_params_seed):
            '''Evaluate all checkpoints (m_fcp) for a single FCP seed against 
            all eval partners (num_eval_total).'''
            # For each checkpoint (m_fcp) we need a set of rng keys for evaluation against all eval partners.
            # rng has shape (2,), rng.val has shape (3,)
            total_rngs = jax.random.split(rng, m_fcp * num_eval_total)
            total_rngs = total_rngs.reshape(m_fcp, num_eval_total, -1)
            # For one checkpoint, evaluate over the partner pool defined by eval_params_flat, 
            # which has shape (num_eval_total, ...)
            def eval_for_checkpoint(fcp_param, rngs_ckpt):
                return jax.vmap(
                    lambda rng, partner_param: eval_pair_fn(rng, fcp_param, partner_param),
                    in_axes=(0, 0)
                )(rngs_ckpt, eval_params_flat)
            # Vmap over the m_fcp checkpoints.
            return jax.vmap(eval_for_checkpoint, in_axes=(0, 0))(fcp_params_seed, total_rngs)

        return jax.vmap(eval_fcp_seed, in_axes=(0, 0))(rngs, fcp_params)

    # --- 6. JIT-compile and run ---
    base_rng = jax.random.PRNGKey(config["EVAL_SEED"])
    # Prepare one RNG per FCP seed (n_fcp total).
    rngs = jax.random.split(base_rng, n_fcp)
    with jax.disable_jit(False):
        eval_metrics = outer_eval(rngs)
    # eval_metrics has shape: (n_fcp, m_fcp, num_eval_total, num_episodes, ...info)
    return eval_metrics

if __name__ == "__main__":
    # set hyperparameters:
    config = {
        "LR": 1.e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": 1e6, # 1e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16, # 4,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.01,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {
        },
        "ANNEAL_LR": True,
        "EVAL_SEED": 12345,
        "NUM_SEEDS": 3,
        "RESULTS_PATH": "results/lbf"
    }

    # training partners
    train_partner_path = "results/lbf/2025-02-13_21-21-35/train_run.pkl"
    partner_ckpts = load_checkpoints(train_partner_path)

    # test_partner_path = "results/lbf/2025-02-13_21-42-20/train_run.pkl"
    # partner_ckpts = load_checkpoints(test_partner_path)

    fcp_path = "results/lbf/2025-02-14_14-22-31/train_run.pkl"
    fcp_ckpts = load_checkpoints(fcp_path)

    print("Starting eval.")
    eval_metrics = eval_fcp_agent(config, fcp_ckpts, partner_ckpts, 
                                  num_episodes=32)
    
    # save eval_metrics to pickle
    with open("results/lbf/2025-02-14_14-22-31/eval_run.pkl", "wb") as f:
        pickle.dump(eval_metrics, f)
    
    # each submetric shape is (num_fcp_seeds, num_fcp_ckpts, num_eval_ckpts, num_rollouts, num_agents)
    # the FCP agent is always agent 0, the partner is agent 1
    print("Mean Return of FCP agent:", eval_metrics["returned_episode_returns"][:, -1,:, :, 0].mean())
    print("Std Return of FCP agent:", eval_metrics["returned_episode_returns"][:, -1,:, :, 0].std())

    print("Mean Percent Eaten of FCP agent:", eval_metrics["percent_eaten"][:, -1,:, :, 0].mean())
    print("Std Percent Eaten of FCP agent:", eval_metrics["percent_eaten"][:, -1,:, :, 0].std())

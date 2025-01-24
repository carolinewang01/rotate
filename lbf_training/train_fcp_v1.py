from ippo_checkpoints import ActorCritic, batchify, unbatchify, Transition
def train_fcp_agent(config, partner_checkpoints):
    """
    Train a single FCP agent (agent_0) against a *frozen* pool of partners (agent_1).
    The partner pool is stored in 'partner_checkpoints', a PyTree where each leaf
    has shape (n_seeds, m_ckpts, ...). 

    This returns a dict with:
      - 'runner_state': final environment and agent state
      - 'metrics': per-update metrics (episode returns, etc.)
      - 'checkpoints': a PyTree of shape (NUM_CHECKPOINTS, ...) in each leaf with FCP snapshots
    matching the style of the original IPPO code.
    """
    import jumanji
    import jaxmarl
    from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState
    import distrax
    import functools

    # 1) Environment creation (identical logic to your IPPO).
    if config["ENV_NAME"] == 'lbf':
        env = jumanji.make('LevelBasedForaging-v0')
        env = JumanjiToJaxMARL(env)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # 2) Define the same ActorCritic net used for agent_0 (trainable).
    # class ActorCritic(nn.Module):
    #     action_dim: int
    #     activation: str = "tanh"

    #     @nn.compact
    #     def __call__(self, x):
    #         if self.activation == "relu":
    #             activation_fn = nn.relu
    #         else:
    #             activation_fn = nn.tanh

    #         # Actor
    #         h = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
    #         h = activation_fn(h)
    #         h = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(h)
    #         h = activation_fn(h)
    #         logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)
    #         pi = distrax.Categorical(logits=logits)

    #         # Critic
    #         v = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(x)
    #         v = activation_fn(v)
    #         v = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(v)
    #         v = activation_fn(v)
    #         value = nn.Dense(1, kernel_init=orthogonal(1.0))(v)

    #         return pi, jnp.squeeze(value, axis=-1)

    # 3) Prepare "partner gather" logic. We assume partner_checkpoints["params"]
    #    has shape [n_seeds, m_ckpts, ...]. We'll pick (seed_idx, ckpt_idx)
    #    at random for each environment. For each environment, we freeze that choice
    #    until the next environment reset or next update (depending on your preference).
    n_seeds = partner_checkpoints["params"]["Dense_0"]["kernel"].shape[0]
    m_ckpts = partner_checkpoints["params"]["Dense_0"]["kernel"].shape[1]
    total_partners = n_seeds * m_ckpts

    def gather_partner_params(params_pytree, flat_idx):
        """
        Given a single integer flat_idx in [0, n_seeds*m_ckpts),
        slice out the partner PyTree from 'params_pytree'.
        """
        seed_idx = flat_idx // m_ckpts
        ckpt_idx = flat_idx % m_ckpts
        return jax.tree_map(lambda x: x[seed_idx, ckpt_idx], params_pytree)

    # 4) Build a training function similar to 'make_train' in ippo_checkpoints.py.
    def make_fcp_train(config):
        # Recompute derived config fields exactly like in IPPO.
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )

        # Learning-rate schedule
        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac

        # The main train(...) function that sets everything up and runs the PPO loop.
        def train(rng):
            # 4a) Init trainable agent_0 network
            network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
            rng, init_rng = jax.random.split(rng)
            dummy_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            init_params = network.init(init_rng, dummy_x)

            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )

            train_state = TrainState.create(
                apply_fn=network.apply,
                params=init_params,
                tx=tx,
            )

            # 4b) Initialize environment & random partner indices
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)

            # For each environment in [0..NUM_ENVS], pick a partner in [0..total_partners).
            rng, partner_rng = jax.random.split(rng)
            partner_indices = jax.random.randint(
                key=partner_rng,
                shape=(config["NUM_ENVS"],),
                minval=0,
                maxval=total_partners,
            )

            # # We replicate the Transition structure & code from IPPO:
            # class Transition(NamedTuple):
            #     done: jnp.ndarray
            #     action: jnp.ndarray
            #     value: jnp.ndarray
            #     reward: jnp.ndarray
            #     log_prob: jnp.ndarray
            #     obs: jnp.ndarray
            #     info: Dict[str, jnp.ndarray]

            # def batchify(x: dict, agent_list, num_actors):
            #     x = jnp.stack([x[a] for a in agent_list])
            #     return x.reshape((num_actors, -1))

            # def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
            #     x = x.reshape((num_agents, num_envs, -1))
            #     return {a: x[i] for i, a in enumerate(agent_list)}

            # 4c) Environment stepping function (collect transitions for agent_0).
            @functools.partial(jax.jit, static_argnums=0)
            def _env_step(runner_state, unused):
                """
                runner_state = (train_state, env_state, last_obs, partner_indices, rng)
                We'll produce a single Transition for agent_0 in each environment.
                """
                (train_state, env_state, last_obs, partner_idx_vec, rng) = runner_state

                # Sample subkeys
                rng, rng_a0, rng_partner, rng_step = jax.random.split(rng, 4)

                # Flatten all agents' obs so we can process agent_0 with the trainable net.
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])  # shape (num_actors, obs_dim)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=rng_a0)
                log_prob = pi.log_prob(action)

                # Next, separate out agent_0's action vs agent_1's action for each environment:
                # shape (num_envs, num_agents=2, obs_dim) => we had batchify with shape (2*num_envs, obs_dim).
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                # env_act is a dict like: { 'agent_0': shape (NUM_ENVS,), 'agent_1': shape (NUM_ENVS,) }

                # Overwrite agent_1's action with the partner policy. We'll do it environment by environment.
                def partner_action_for_env(i, carry):
                    # Gather partner params:
                    p_idx = partner_idx_vec[i]
                    p_params = gather_partner_params(partner_checkpoints["params"], p_idx)
                    # Flatten obs for agent_1 in env i:
                    obs_1 = last_obs[i]["agent_1"].flatten()
                    # Forward pass:
                    partner_pi, _ = network.apply(p_params, obs_1)
                    # Sample action for agent_1
                    a1 = partner_pi.sample(seed=jax.random.fold_in(rng_partner, i))
                    return carry, a1

                # We'll do a jax.lax.scan to get all agent_1 actions across envs:
                _, partner_actions = jax.lax.scan(partner_action_for_env, None, jnp.arange(config["NUM_ENVS"]))
                # Now place those actions in env_act["agent_1"]:
                env_act["agent_1"] = partner_actions

                # Step the environment in parallel:
                step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    step_rngs, env_state, env_act
                )

                # Flatten info dict to shape (NUM_ACTORS, any_info_dim)
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                # Build transitions for agent_0 specifically (the first agent in batchify ordering).
                transition = Transition(
                    done=batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                )

                new_runner_state = (train_state, env_state, obsv, partner_idx_vec, rng)
                return new_runner_state, transition

            # 4d) GAE advantage (same as IPPO).
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            # 4e) PPO updates (same as IPPO).
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, returns = batch_info

                def _loss_fn(params, traj_batch, gae, target_v):
                    pi, value = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_loss_1 = jnp.square(value - target_v)
                    value_loss_2 = jnp.square(value_pred_clipped - target_v)
                    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_1, value_loss_2))

                    # Policy gradient loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae_norm
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae_norm
                    )
                    loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

                    # Entropy
                    entropy = jnp.mean(pi.entropy())

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss_val, aux_vals), grads = grad_fn(
                    train_state.params, traj_batch, advantages, returns
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, (loss_val, aux_vals)

            def _update_epoch(update_state, unused):
                train_state, traj_batch, advantages, returns, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch_size = config["NUM_STEPS"] * config["NUM_ACTORS"]
                permutation = jax.random.permutation(perm_rng, batch_size)

                # Flatten all
                def flatten(x):
                    return x.reshape((batch_size,) + x.shape[2:])

                traj_batch_flat = jax.tree_util.tree_map(flatten, traj_batch)
                advantages_flat = advantages.reshape((batch_size,))
                returns_flat = returns.reshape((batch_size,))

                # Shuffle
                traj_batch_shuf = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    traj_batch_flat
                )
                advantages_shuf = jnp.take(advantages_flat, permutation, axis=0)
                returns_shuf = jnp.take(returns_flat, permutation, axis=0)

                # Reshape into minibatches
                def reshape_mb(x):
                    return x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:])

                traj_mb = jax.tree_util.tree_map(reshape_mb, traj_batch_shuf)
                adv_mb = advantages_shuf.reshape((config["NUM_MINIBATCHES"], -1))
                ret_mb = returns_shuf.reshape((config["NUM_MINIBATCHES"], -1))

                def _scan_minibatches(carry, idx):
                    train_state = carry
                    mb_data = (
                        jax.tree_util.tree_map(lambda z: z[idx], traj_mb),
                        adv_mb[idx],
                        ret_mb[idx],
                    )
                    train_state, _ = _update_minbatch(train_state, mb_data)
                    return train_state, ()

                train_state, _ = jax.lax.scan(_scan_minibatches, train_state, jnp.arange(config["NUM_MINIBATCHES"]))
                return (train_state, traj_batch, advantages, returns, rng), ()

            # 4f) Full PPO update step (collect rollout + advantage + multiple epochs),
            #     then re-sample partner indices if desired.
            def _update_step(update_runner_state, unused):
                (runner_state, update_steps) = update_runner_state
                train_state, env_state_, obs_, partner_idx_vec, rng = runner_state

                # Rollout via scan
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
                # Unpack
                train_state, env_state, last_obs, partner_idx_vec, rng = runner_state

                # Compute final value
                obs_batch_final = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                _, last_val = network.apply(train_state.params, obs_batch_final)

                # GAE
                advantages, returns = _calculate_gae(traj_batch, last_val)

                # PPO epochs
                update_state = (train_state, traj_batch, advantages, returns, rng)
                update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
                train_state = update_state[0]

                # (Optional) Re-sample partner indices so next update sees new partners
                rng, idx_rng = jax.random.split(rng)
                new_partner_indices = jax.random.randint(
                    key=idx_rng,
                    shape=(config["NUM_ENVS"],),
                    minval=0,
                    maxval=total_partners,
                )

                # Collect metrics from the last rollout batch
                # e.g. "returned_episode_returns", "percent_eaten", etc. 
                # (Same pattern as in IPPO code):
                metric = traj_batch.info
                metric["update_steps"] = update_steps

                # Put updated runner state & increment
                new_runner_state = (train_state, env_state, last_obs, new_partner_indices, rng)
                return ((new_runner_state, update_steps + 1), metric)

            # 4g) Checkpoint logic: identical to IPPO. We'll store (NUM_CHECKPOINTS, ...) snapshots.
            checkpoint_interval = max(1, config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"])
            num_ckpts = config["NUM_CHECKPOINTS"]

            def init_ckpt_array(params_pytree):
                # Create a param array of shape (num_ckpts,) + param.shape for each leaf
                return jax.tree_map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree
                )

            def _update_step_with_checkpoint(update_with_ckpt_state, unused):
                (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_state
                (update_runner_state, metric) = _update_step(update_runner_state, None)
                # update_runner_state = ((train_state, env_state, obs, partner_idx_vec, rng), update_steps)
                runner_substate, update_steps = update_runner_state

                # Save checkpoint if we hit the interval
                to_store = jnp.equal(jnp.mod(update_steps, checkpoint_interval), 0)

                def store_ckpt_fn(args):
                    _checkpoint_array, _ckpt_idx = args
                    train_state = runner_substate[0]
                    new_ckpt_array = jax.tree_map(
                        lambda arr, p: arr.at[_ckpt_idx].set(p),
                        _checkpoint_array,
                        train_state.params
                    )
                    return (new_ckpt_array, _ckpt_idx + 1)

                def skip_ckpt_fn(args):
                    return args

                (checkpoint_array, ckpt_idx) = jax.lax.cond(
                    to_store,
                    store_ckpt_fn,
                    skip_ckpt_fn,
                    (checkpoint_array, ckpt_idx)
                )
                return ((update_runner_state, checkpoint_array, ckpt_idx), metric)

            # 4h) Run the main update loop via jax.lax.scan
            rng, loop_rng = jax.random.split(rng)
            update_steps = 0
            runner_state = (train_state, env_state, obsv, partner_indices, loop_rng)
            update_runner_state = (runner_state, update_steps)

            checkpoint_array = init_ckpt_array(train_state.params)
            ckpt_idx = 0
            update_with_ckpt_state = (update_runner_state, checkpoint_array, ckpt_idx)

            update_with_ckpt_state, metrics = jax.lax.scan(
                _update_step_with_checkpoint,
                update_with_ckpt_state,
                xs=None,
                length=config["NUM_UPDATES"]
            )

            # Final runner_state, final checkpoints
            (final_update_runner_state, final_checkpoint_array, final_ckpt_idx) = update_with_ckpt_state
            final_train_state, _ = final_update_runner_state

            # Slice down if we wrote fewer than num_ckpts
            final_checkpoints = jax.tree_map(
                lambda arr: arr[:final_ckpt_idx],
                final_checkpoint_array
            )

            return {
                "runner_state": final_update_runner_state,
                "metrics": metrics,  # shape (NUM_UPDATES, ...), each entry has your tracked info
                "checkpoints": final_checkpoints,
            }

        return train

    # 5) Build our training function and run it
    train_fcp_fn = make_fcp_train(config)
    rng = jax.random.PRNGKey(config["SEED"])

    # If you want multiple FCP agents for multiple seeds:
    #   rngs = jax.random.split(rng, config["NUM_SEEDS"])
    #   out = jax.vmap(train_fcp_fn)(rngs)
    # else just do single-seed training:
    out = train_fcp_fn(rng)

    return out["checkpoints"], out["metrics"]

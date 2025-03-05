def train_fcp_agent(config, checkpoints):
    '''
    Train an FCP agent using the given partner checkpoints and IPPO.
    Return model checkpoints and metrics. 
    '''
    # ------------------------------
    # 1) Flatten partner checkpoints into shape (N, ...) if desired
    #    but we can also keep them as (n_seeds, m_ckpts, ...).
    #    We'll just do gather via dynamic indexing in a jittable way.
    # ------------------------------
    partner_params = checkpoints["params"]  # This is the full PyTree
    n_seeds, m_ckpts = partner_params["Dense_0"]["kernel"].shape[:2]
    num_total_partners = n_seeds * m_ckpts

    # We can define a small helper to gather the correct slice for each environment
    # from shape (n_seeds, m_ckpts, ...) -> (num_envs, ...)
    # We'll do an integer mapping from [0, num_total_partners) -> (seed_idx, ckpt_idx).
    def unravel_partner_idx(idx):
        seed_idx = idx // m_ckpts
        ckpt_idx = idx % m_ckpts
        return seed_idx, ckpt_idx

    def gather_partner_params(partner_params_pytree, idx_vec):
        # Function implementation remains the same
        pass

    # ------------------------------
    # 3) Build the FCP training function, closely mirroring `make_train(...)`.
    # ------------------------------
    def make_fcp_train(config, partner_params):
        env = make_env(config)
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        
        env = LogWrapper(env)

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac

        def train(rng):
            # INIT NETWORKS
            # Use ActorCriticRNN for agent_0_net (ego agent)
            agent_0_net = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
            agent_1_net = ActorCritic(env.action_space(env.agents[1]).n)
            
            rng, _rng = jax.random.split(rng)
            
            # Initialize ego agent (RNN)
            init_x = (
                jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
                jnp.zeros((1, config["NUM_ENVS"])),
                jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            )
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            agent_0_params = agent_0_net.init(_rng, init_hstate, init_x)
            
            # Initialize partner agent (standard network)
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((1, env.observation_space(env.agents[1]).shape[0]))
            agent_1_params = agent_1_net.init(_rng, init_x)
            
            # Setup optimizers
            if config["ANNEAL_LR"]:
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule),
                )
            else:
                tx_0 = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"]),
                )
            
            # Initialize TrainState for the ego agent
            train_state_0 = TrainState.create(
                apply_fn=agent_0_net.apply,
                params=agent_0_params,
                tx=tx_0,
            )

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            
            # Initialize hidden state for RNN
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            
            # TRAIN LOOP
            def _update_step(runner_state, unused):
                train_state_0, env_state, last_obs, last_done, hstate, rng = runner_state
                
                # COLLECT TRAJECTORIES
                def _env_step(step_state, unused):
                    train_state_0, env_state, last_obs, last_done, hstate, rng = step_state
                    
                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    
                    # Prepare inputs for agent 0 (RNN)
                    avail_actions_0 = jnp.ones((config["NUM_ENVS"], env.action_space(env.agents[0]).n))
                    # Input shape for RNN: (batch, sequence_length, features)
                    rnn_input = (
                        last_obs[env.agents[0]].reshape(1, config["NUM_ENVS"], -1),  # obs
                        last_done.reshape(1, config["NUM_ENVS"]),  # done
                        avail_actions_0.reshape(1, config["NUM_ENVS"], -1)  # avail_actions
                    )
                    
                    # Get values, actions, log_probs, and new hidden state for agent 0
                    (pi_0, value_0, new_hstate) = agent_0_net.apply(train_state_0.params, hstate, rnn_input)
                    action_0 = pi_0.sample(seed=_rng)
                    log_prob_0 = pi_0.log_prob(action_0)
                    
                    # Handle partner agent action selection using partner network
                    # (Implementation remains the same)
                    # ...
                    
                    # STEP ENV
                    # (Implementation remains the same)
                    # ...
                    
                    # Ensure that done flags are passed to reset the RNN hidden state
                    transition = Transition(
                        global_done=global_done,
                        done=done,
                        action=action,
                        value=value,
                        reward=reward,
                        log_prob=log_prob,
                        obs=last_obs,
                        info=info,
                        avail_actions=avail_actions,
                    )
                    
                    next_state = (train_state_0, env_state, obs, done, new_hstate, rng)
                    return next_state, transition
                
                # Run rollout
                rng, _rng = jax.random.split(rng)
                (train_state_0, env_state, last_obs, last_done, hstate, rng), transitions = jax.lax.scan(
                    _env_step, (train_state_0, env_state, last_obs, last_done, hstate, rng), None, config["NUM_STEPS"]
                )
                
                # CALCULATE ADVANTAGE
                # (Implementation remains mostly the same, but handle RNN properly)
                # ...
                
                # RNN-specific: For value bootstrapping, we need to use the RNN to get the value
                avail_actions_0 = jnp.ones((config["NUM_ENVS"], env.action_space(env.agents[0]).n))
                rnn_input = (
                    last_obs[env.agents[0]].reshape(1, config["NUM_ENVS"], -1),
                    last_done.reshape(1, config["NUM_ENVS"]),
                    avail_actions_0.reshape(1, config["NUM_ENVS"], -1)
                )
                _, last_val_0, _ = agent_0_net.apply(train_state_0.params, hstate, rnn_input)
                
                # Rest of advantage calculation remains the same
                # ...
                
                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    # (Implementation needs to be modified for RNN)
                    # ...
                    
                    def _update_minibatch(train_state_0, batch_info):
                        # Extract minibatch
                        mb_obs, mb_actions, mb_advantages, mb_returns, mb_log_probs, mb_dones, mb_avail = batch_info
                        
                        # Define loss function that incorporates RNN processing
                        def _loss_fn(params_0, obs_0, actions_0, advantages_0, returns_0, log_probs_0, dones_0, avail_0):
                            # Initialize hidden state
                            batch_size = obs_0.shape[0] // config["NUM_STEPS"]
                            init_h = ScannedRNN.initialize_carry(batch_size, config["GRU_HIDDEN_DIM"])
                            
                            # Process inputs for RNN
                            seq_len = config["NUM_STEPS"]
                            obs_seq = obs_0.reshape(seq_len, batch_size, -1)
                            done_seq = dones_0.reshape(seq_len, batch_size)
                            avail_seq = avail_0.reshape(seq_len, batch_size, -1)
                            rnn_input = (obs_seq, done_seq, avail_seq)
                            
                            # Get outputs from RNN
                            pi_0, values_0, _ = agent_0_net.apply(params_0, init_h, rnn_input)
                            
                            # Flatten outputs to match other tensors
                            values_0 = values_0.reshape(-1)
                            log_probs_new_0 = pi_0.log_prob(actions_0)
                            
                            # Calculate losses using standard PPO objective
                            # (Remainder of loss calculation remains the same)
                            # ...
                            
                            return loss_0, (pi_loss_0, vf_loss_0, entropy_loss_0)
                        
                        # Compute gradients and update
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        (loss_0, (pi_loss_0, vf_loss_0, entropy_loss_0)), grads_0 = grad_fn(
                            train_state_0.params, 
                            mb_obs[env.agents[0]], mb_actions[env.agents[0]], 
                            mb_advantages[env.agents[0]], mb_returns[env.agents[0]], 
                            mb_log_probs[env.agents[0]], mb_dones, mb_avail[env.agents[0]]
                        )
                        
                        train_state_0 = train_state_0.apply_gradients(grads=grads_0)
                        return train_state_0, (loss_0, pi_loss_0, vf_loss_0, entropy_loss_0)
                    
                    # Scan across minibatches
                    # ...
                    
                    return update_state, loss_info
                
                # Scan across update epochs
                # ...
                
                return (train_state_0, env_state, last_obs, last_done, hstate, rng), metrics
            
            # Scan across updates
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state_0, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng)
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
            
            # Extract final values
            train_state_0 = runner_state[0]
            metrics = jax.tree_map(lambda x: x.swapaxes(0, 1), metrics)
            
            # Return dictionary with necessary information
            return {
                "params": train_state_0.params,
                "metrics": metrics,
            }
        
        return train
    
    # ------------------------------
    # 4) Actually run the FCP training
    # ------------------------------
    # Rest of the implementation remains the same
    # ...

# Make sure to add any missing RNN-related configuration parameters
if __name__ == "__main__":
    # set hyperparameters:
    config = {
        # Original parameters...
        "LR": 1.e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": 3e5,
        "UPDATE_EPOCHS": 15,
        "NUM_MINIBATCHES": 16,
        "NUM_CHECKPOINTS": 5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.05,
        "ENT_COEF": 0.01,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "ENV_NAME": "lbf",
        "ENV_KWARGS": {},
        "SEED": 38410, 
        "PARTNER_SEED": 112358,
        "NUM_SEEDS": 3,
        "RESULTS_PATH": "results/lbf",
        
        # Add RNN-specific parameters
        "GRU_HIDDEN_DIM": 128,  # Hidden dimension for GRU
    }
    
    # Rest of the main function remains the same
    # ...
import jax
from envs import make_env
from evaluation.policy_loaders import MLPActorCriticPolicyLoader, S5ActorCriticPolicyLoader, RandomActor


def rollout(ego_run_path, partner_run_path, 
            ego_seed_idx, partner_seed_idx,
            ego_checkpoint_idx, partner_checkpoint_idx,
            num_episodes, render, 
            savevideo, save_name
            ):
    env = make_env('lbf')

    policies = {}
    policies[0] = S5ActorCriticPolicyLoader(ego_run_path, env.action_spaces['agent_0'].n, 
                                            n=ego_seed_idx, m=ego_checkpoint_idx)
    policies[1] = MLPActorCriticPolicyLoader(partner_run_path, env.action_spaces['agent_1'].n, 
                                             n=partner_seed_idx, m=partner_checkpoint_idx) 

    # Rollout
    states = []
    key = jax.random.PRNGKey(112358)

    for episode in range(NUM_EPISODES):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)

        done = {agent: False for agent in env.agents}
        done['__all__'] = False
        total_rewards = {agent: 0.0 for agent in env.agents}
        num_steps = 0
        while not done['__all__']:
            # Sample actions for each agent
            actions = {}
            for i, agent in enumerate(env.agents):
                # Policies tend to perform better on LBF in train mode
                action, key = policies[i].act(obs[agent], key, test_mode=False)
                actions[agent] = action
            
            key, subkey = jax.random.split(key)
            obs, state, rewards, done, info = env.step(subkey, state, actions)

            # Process observations, rewards, dones, and info as needed
            for agent in env.agents:
                total_rewards[agent] += rewards[agent]
                print("action is ", actions[agent])
                print("obs", obs[agent], "type", type(obs[agent]))
                print("rewards", rewards[agent], "type", type(rewards[agent]))
                print("dones", done[agent], "type", type(done[agent]))
                print("info", info, "type", type(info))

            num_steps += 1        
            if RENDER:         
                env.render(state)
                states.append(state)
        print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")
        
    if RENDER and SAVEVIDEO:
        anim = env.animate(states, interval=150)
        anim.save("results/lbf/gifs/{save_name}.gif", 
                  writer="imagemagick")

if __name__ == "__main__":
    NUM_EPISODES = 4
    RENDER = True
    SAVEVIDEO = False
    
    ego_run_path = "results/lbf/2025-02-17_14-38-26/train_run.pkl" # FCP agent, trained for 3e6 steps
    partner_run_path = "results/lbf/2025-02-13_21-21-35/train_run.pkl" # FCP training partner, trained for 3e6 steps
    save_name = "fcp=2025-02-13_21-21-35_partner=2025-02-17_14-38-26"
    
    rollout(env_id="lbf", 
            ego_run_path=ego_run_path, 
            partner_run_path=partner_run_path, 
            num_episodes=NUM_EPISODES, 
            render=RENDER, 
            savevideo=SAVEVIDEO,
            save_name=save_name
            )

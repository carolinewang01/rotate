import jax
from fcp.networks import ActorCritic
from fcp.utils import load_checkpoints, make_env

def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class ActorCriticPolicyWrapper():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''
        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.model = ActorCritic(action_dim)

    def act(self, obs, rng, test_mode=True):
        '''Returns an action given an observation.'''
        rng, act_rng = jax.random.split(rng)        
        pi, _ = self.model.apply(self.model_params, obs)
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        return action, rng

class RandomActor():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, rng, test_mode=True):
        rng, act_rng = jax.random.split(rng)
        action = int(self.action_space.sample(act_rng))
        return action, rng


if __name__ == "__main__":

    NUM_EPISODES = 4
    RENDER = True
    SAVEVIDEO = False

    # Instantiate a Jumanji environment using the registry
    # env = jumanji.make('LevelBasedForaging-v0')
    # env = JumanjiToJaxMARL(env)
    env = make_env('lbf')

    # Instantiate a policy
    ego_run_path = "results/lbf/2025-02-17_14-38-26/train_run.pkl" # FCP agent, trained for 3e6 steps
    partner_run_path = "results/lbf/2025-02-13_21-21-35/train_run.pkl" # FCP training partner, trained for 3e6 steps
    
    policies = {}
    policies[0] = ActorCriticPolicyWrapper(ego_run_path, env.action_spaces['agent_0'].n, n=0, m=-1)
    policies[1] = ActorCriticPolicyWrapper(partner_run_path, env.action_spaces['agent_1'].n, n=0, m=0) 
    # policies[1] = RandomActor(env.action_space("agent_1"))

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
                action, key = policies[i].act(obs[agent], key, test_mode=False)
                actions[agent] = action
            
            key, subkey = jax.random.split(key)
            obs, state, rewards, done, info = env.step(subkey, state, actions)

            # Process observations, rewards, dones, and info as needed
            for agent in env.agents:
                total_rewards[agent] += rewards[agent]
                print("action is ", actions[agent])
                # print("obs", obs[agent], "type", type(obs[agent]))
                print("rewards", rewards[agent], "type", type(rewards[agent]))
                print("dones", done[agent], "type", type(done[agent]))
                print("info", info, "type", type(info))
                # print("avail actions are ", env.get_avail_actions(state.env_state)[agent])

            num_steps += 1        
            if RENDER:         
                env.render(state)
                states.append(state)
        print(f"Episode {episode} finished. Total rewards: {total_rewards}. Num steps: {num_steps}")

        # At end of episode, freeze the terminal frame to pause the GIF.
        if RENDER:
            for _ in range(3):
                states.append(state)
        
    if RENDER and SAVEVIDEO:
        anim = env.animate(states, interval=150)
        anim.save("figures/lbf_fcp=2025-02-13_21-21-35_partner=2025-02-17_14-38-26.gif", 
                  writer="imagemagick")
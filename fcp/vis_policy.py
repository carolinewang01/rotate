import jax
import jumanji
from jumanji import specs
import matplotlib.pyplot as plt
import pickle

from fcp.networks import ActorCritic
from fcp.utils import load_checkpoints
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL


def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class ActorCriticPolicyWrapper():
    def __init__(self, train_run_path, action_dim):
        policy_checkpoints = load_checkpoints(train_run_path)
        n, m = 0, -1  # select last checkpoint of first seed
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
    
if __name__ == "__main__":

    NUM_EPISODES = 1
    RENDER = True
    SAVEVIDEO = False

    # Instantiate a Jumanji environment using the registry
    env = jumanji.make('LevelBasedForaging-v0')
    env = JumanjiToJaxMARL(env)

    # Instantiate a policy
    run_path = "results/lbf/2025-02-13_21-21-35/train_run.pkl" # FCP training partner, trained for 3e6 steps
    policy = ActorCriticPolicyWrapper(run_path, env.action_spaces['agent_0'].n)

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
            for agent in env.agents:
                action, key = policy.act(obs[agent], key, test_mode=False)
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
        anim.save("figures/lbf.gif", writer="imagemagick")
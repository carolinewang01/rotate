import jax
from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic
from common.save_load_utils import load_checkpoints


def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class MLPActorCriticPolicyLoader():
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

class S5ActorCriticPolicyLoader():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''
        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.model = S5ActorCritic(action_dim) # TODO: check if input params are correct
        # TODO: initialize any hidden states needed
        # TODO: maintain hidden state internally?

    def act(self, obs, rng, test_mode=True):
        '''Returns an action given an observation.'''
        rng, act_rng = jax.random.split(rng)        
        # TODO: construct obs dict
        # TODO: maintain hidden state
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

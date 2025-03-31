import jax
from common.mlp_actor_critic import ActorCritic
from common.s5_actor_critic import S5ActorCritic
from common.save_load_utils import load_checkpoints
import jax.numpy as jnp
from common.s5_ssm import make_DPLR_HiPPO, init_S5SSM


def select_checkpoint_params(full_checkpoints, seed_idx, ckpt_idx):
    """
    Slices the pytree so each leaf contains only the parameters for
    the given seed and checkpoint.
    """
    # 'full_checkpoints' is the entire dict with top-level keys like 'params', etc.
    return jax.tree.map(lambda x: x[seed_idx, ckpt_idx], full_checkpoints)

class MLPActorCriticLoader():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        self.model = ActorCritic(action_dim)

    def act(self, obs_dict, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)        
        pi, _ = self.model.apply(self.model_params, (obs_dict['obs'], obs_dict['avail_actions'])) 
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        return action, rng

class S5ActorCriticLoader():
    def __init__(self, train_run_path, action_dim, n=0, m=-1):
        '''Loads and initializes a single policy checkpoint from saved pickled file.

        train_run_path: path to the training run pickle file
        action_dim: number of actions
        n: seed index
        m: checkpoint index
        '''
        policy_checkpoints = load_checkpoints(train_run_path)
        self.model_params = select_checkpoint_params(policy_checkpoints, n, m)
        
        # Initialize S5 config
        # TODO: avoid hard-coding SSM configs in future
        self.config = {
            "S5_D_MODEL": 16,
            "S5_N_LAYERS": 2,
            "S5_ACTIVATION": "full_glu",
            "S5_DO_NORM": True,
            "S5_PRENORM": True,
            "S5_DO_GTRXL_NORM": True,
            "S5_BLOCKS": 1,
            "S5_SSM_SIZE": 16,
            "S5_ACTOR_CRITIC_HIDDEN_DIM": 64,
        }
        
        # Initialize S5 SSM
        Lambda, P, B, V, B_orig = make_DPLR_HiPPO(self.config["S5_SSM_SIZE"])
        ssm_init_fn = init_S5SSM(
            H=self.config["S5_D_MODEL"],
            P=self.config["S5_SSM_SIZE"],
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=V.conj().T)
        
        self.model = S5ActorCritic(
            action_dim=action_dim,
            config=self.config,
            ssm_init_fn=ssm_init_fn,
            fc_hidden_dim=self.config["S5_ACTOR_CRITIC_HIDDEN_DIM"],
            ssm_hidden_dim=self.config["S5_SSM_SIZE"]
        )
        
        # Initialize hidden state
        self.hidden = self.model.s5.initialize_carry(batch_size=1, hidden_size=self.config["S5_D_MODEL"], n_layers=self.config["S5_N_LAYERS"])

    def act(self, obs_dict, rng, test_mode=True):
        '''Returns an action given an observation.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)
        
        # Update hidden state and get action distribution
        self.hidden, pi, _ = self.model.apply(self.model_params, self.hidden, obs_dict)
        
        if test_mode:
            action = pi.mode()
        else:
            action = pi.sample(seed=act_rng)
        return action, rng

class RandomActor():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs_dict, rng, test_mode=True):
        '''Returns a random action from available actions.
        obs_dict: dict containing 'obs', 'dones', and 'avail_actions'
        '''
        rng, act_rng = jax.random.split(rng)
        avail_actions = obs_dict['avail_actions']
        # Get indices of available actions
        avail_indices = jnp.where(avail_actions == 1)[0]
        # Sample from available actions
        action = int(jax.random.choice(act_rng, avail_indices))
        return action, rng

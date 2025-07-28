from jax import random as jr, numpy as jnp
import gymnax

class Discretizer:
    """Handles discretization of continuous spaces."""
    def __init__(self, num_bins, low, high):
        self.num_bins = num_bins
        self.low = low
        self.high = high
        
    def discretize(self, value):
        """Convert continuous value to discrete bin index."""
        normalized = (value - self.low) / (self.high - self.low)
        indices = jnp.floor(normalized * self.num_bins).astype(jnp.int32)
        return jnp.clip(indices, 0, self.num_bins - 1)
        
    def undiscretize(self, indices):
        """Convert discrete indices to continuous values."""
        normalized = (indices + 0.5) / self.num_bins
        return self.low + normalized * (self.high - self.low)


class EnvWrapper:
    """Base wrapper class for Gymnax environments."""
    def __init__(self, env, env_params, num_bins=10, seed=0):
        self.env = env
        self.env_params = env_params
        self.num_bins = num_bins
        self.key = jr.PRNGKey(seed)
        self.name = "BaseEnv"
        
    def reset(self, key):
        """Reset the environment."""
        obs, state = self.env.reset(key, self.env_params)
        return self.process_observation(obs), state
        
    def step(self, key, state, action):
        """Take a step in the environment."""
        env_action = self.process_action(action)
        next_obs, next_state, reward, done, info = self.env.step(
            key, state, env_action, self.env_params
        )
        return self.process_observation(next_obs), next_state, reward, done, info
        
    def process_observation(self, obs):
        """Process the observation. To be implemented by subclasses."""
        raise NotImplementedError
        
    def process_action(self, action):
        """Process the action."""
        raise NotImplementedError
        
    def get_observation_space(self):
        """Return information about the observation space."""
        raise NotImplementedError
        
    def get_action_space(self):
        """Return information about the action space."""
        raise NotImplementedError
    
    def discretize_state(self, state):
        """Discretize a state. To be implemented by subclasses."""
        raise NotImplementedError


class PendulumWrapper(EnvWrapper):
    """Wrapper for Pendulum-v1 environment."""
    def __init__(self, num_bins=10, seed=0):
        env, env_params = gymnax.make("Pendulum-v1")
        super().__init__(env, env_params, num_bins, seed)
        self.name = "Pendulum-v1"
        
        # state space bounds for Pendulum
        self.obs_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.state_low = jnp.array([-1.0, -1.0, -8.0])
        self.state_high = jnp.array([1.0, 1.0, 8.0])
        
        # action space bounds for Pendulum
        self.action_low = -2.0
        self.action_high = 2.0
        
        self.state_discretizer = Discretizer(num_bins, self.state_low, self.state_high)
        self.action_discretizer = Discretizer(
            num_bins, 
            jnp.array([self.action_low]), 
            jnp.array([self.action_high])
        )
        
        self.discrete_actions = jnp.linspace(
            self.action_low, self.action_high, num_bins
        )
        
    def process_observation(self, obs):
        """Process the observation."""
        return obs
        
    def process_action(self, action_idx):
        """Convert discrete action index to continuous action."""
        if isinstance(action_idx, (int, jnp.integer)):
            return jnp.array([self.discrete_actions[action_idx]])
        return action_idx
        
    def get_observation_space(self):
        """Return information about the observation space."""
        return {
            'dim': self.obs_dim,
            'low': self.state_low,
            'high': self.state_high,
            'num_bins': self.num_bins
        }
        
    def get_action_space(self):
        """Return information about the action space."""
        return {
            'dim': 1,
            'low': self.action_low,
            'high': self.action_high,
            'num_bins': self.num_bins,
            'discrete_actions': self.discrete_actions,
            'is_discrete': False
        }
        
    def discretize_state(self, state):
        """Discretize a state."""
        return self.state_discretizer.discretize(state)
        
    def discretize_action(self, action):
        """Find the index of the closest discrete action."""
        if isinstance(action, float) or jnp.isscalar(action):
            return jnp.argmin(jnp.abs(self.discrete_actions - action))
        return jnp.argmin(jnp.abs(self.discrete_actions - action[0]))


class CartPoleWrapper(EnvWrapper):
    """Wrapper for CartPole-v1 environment."""
    def __init__(self, num_bins=10, seed=0):
        env, env_params = gymnax.make("CartPole-v1")
        super().__init__(env, env_params, num_bins, seed)
        self.name = "CartPole-v1"
        
        # state space bounds for CartPole
        self.obs_dim = 4  # [cart position, cart velocity, pole angle, pole velocity]
        self.state_low = jnp.array([-4.8, -5.0, -0.418, -5.0])
        self.state_high = jnp.array([4.8, 5.0, 0.418, 5.0])
        
        # action space for CartPole is discrete (0 or 1)
        self.action_dim = 2
        
        self.state_discretizer = Discretizer(num_bins, self.state_low, self.state_high)
        
    def process_observation(self, obs):
        """Process the observation."""
        return obs
        
    def process_action(self, action_idx):
        """Convert discrete action index to environment action."""
        return action_idx
        
    def get_observation_space(self):
        """Return information about the observation space."""
        return {
            'dim': self.obs_dim,
            'low': self.state_low,
            'high': self.state_high,
            'num_bins': self.num_bins
        }
        
    def get_action_space(self):
        """Return information about the action space."""
        return {
            'dim': 1,
            'num_bins': self.action_dim,
            'is_discrete': True
        }
        
    def discretize_state(self, state):
        """Discretize a state."""
        return self.state_discretizer.discretize(state)
        
    def discretize_action(self, action):
        """For CartPole, action is already discrete."""
        return action


class MountainCarWrapper(EnvWrapper):
    """Wrapper for MountainCar-v0 environment."""
    def __init__(self, num_bins=10, seed=0):
        env, env_params = gymnax.make("MountainCar-v0")
        super().__init__(env, env_params, num_bins, seed)
        self.name = "MountainCar-v0"
        
        # state space bounds for MountainCar
        self.obs_dim = 2  # [position, velocity]
        self.state_low = jnp.array([-1.2, -0.07])
        self.state_high = jnp.array([0.6, 0.07])
        
        # action space for MountainCar is discrete (0, 1, 2)
        self.action_dim = 3
        
        self.state_discretizer = Discretizer(num_bins, self.state_low, self.state_high)
        
    def process_observation(self, obs):
        """Process the observation."""
        return obs
        
    def process_action(self, action_idx):
        """Convert discrete action index to environment action."""
        return action_idx
        
    def get_observation_space(self):
        """Return information about the observation space."""
        return {
            'dim': self.obs_dim,
            'low': self.state_low,
            'high': self.state_high,
            'num_bins': self.num_bins
        }
        
    def get_action_space(self):
        """Return information about the action space."""
        return {
            'dim': 1,
            'num_bins': self.action_dim,
            'is_discrete': True
        }
        
    def discretize_state(self, state):
        """Discretize a state."""
        return self.state_discretizer.discretize(state)
        
    def discretize_action(self, action):
        """For MountainCar, action is already discrete."""
        return action


class EnvFactory:
    """Factory for creating environment wrappers."""
    @staticmethod
    def create(env_name, num_bins=10, seed=0):
        """Create an environment wrapper."""
        if env_name.lower() == "pendulum":
            return PendulumWrapper(num_bins, seed)
        elif env_name.lower() == "cartpole":
            return CartPoleWrapper(num_bins, seed)
        elif env_name.lower() == "mountaincar":
            return MountainCarWrapper(num_bins, seed)
        else:
            raise ValueError(f"Unknown environment: {env_name}")

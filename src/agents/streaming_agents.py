import jax
from jax import random as jr, numpy as jnp
import gymnax

class StreamingAgent:
    """Base class for streaming reinforcement learning agents."""
    def __init__(self, env_wrapper):
        self.env_wrapper = env_wrapper


class BasicStreamQAgent(StreamingAgent):
    """
    Streaming Q-learning agent with discretized state space.
    """
    def __init__(
        self,
        env_wrapper,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        seed=0
    ):
        super().__init__(env_wrapper)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.key = jr.PRNGKey(seed)
        
        obs_space = env_wrapper.get_observation_space()
        act_space = env_wrapper.get_action_space()
        
        self.obs_dim = obs_space['dim']
        self.num_bins = obs_space['num_bins']
        
        if act_space.get('is_discrete', False):
            self.action_is_discrete = True
            self.num_actions = act_space['num_bins']
        else:
            self.action_is_discrete = False
            self.discrete_actions = act_space['discrete_actions']
            self.num_actions = len(self.discrete_actions)
        
        q_table_shape = tuple([self.num_bins] * self.obs_dim + [self.num_actions])
        self.q_table = jnp.zeros(q_table_shape)
    
    def discretize_state(self, state):
        """Convert continuous state to discrete indices for Q-table lookup."""
        indices = self.env_wrapper.discretize_state(state)
        return tuple(indices)
    
    def select_action(self, state):
        """Epsilon-greedy action selection based on current Q-values."""
        self.key, subkey = jr.split(self.key)
        
        state_idx = self.discretize_state(state)
        
        # epsilon-greedy selection
        if jr.uniform(subkey) < self.epsilon:
            self.key, subkey = jr.split(self.key)
            action_idx = jr.randint(subkey, (), 0, self.num_actions)
        else:
            action_idx = jnp.argmax(self.q_table[state_idx])
        
        return action_idx
    
    def update_q_table(self, state, action_idx, reward, next_state, done):
        """Update Q-value using the Q-learning update rule."""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        current_q = self.q_table[state_idx][action_idx]
        next_max_q = jnp.max(self.q_table[next_state_idx]) if not done else 0.0
        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q
        
        new_q = current_q + self.alpha * td_error
        
        idx = state_idx + (action_idx,)
        self.q_table = self.q_table.at[idx].set(new_q)
        
        return td_error
    
    def run(self, num_steps):
        """Run the agent for the specified number of steps."""
        self.key, key_reset = jr.split(self.key)
        obs, state = self.env_wrapper.reset(key_reset)
        
        results = []
        total_reward = 0
        
        for step in range(num_steps):
            action_idx = self.select_action(obs)
            
            self.key, key_step = jr.split(self.key)
            env_action = self.env_wrapper.process_action(action_idx)
            
            next_obs, next_state, reward, done, info = self.env_wrapper.step(
                key_step, state, env_action
            )
            
            self.update_q_table(obs, action_idx, reward, next_obs, done)
            
            results.append((obs, env_action, reward, done))
            total_reward += reward
            
            obs, state = next_obs, next_state
            
            if done:
                break
                
        return results, total_reward, step + 1

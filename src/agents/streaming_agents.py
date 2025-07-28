import jax
from jax import random as jr, numpy as jnp
import gymnax
from tqdm import tqdm
import equinox as eqx
import optax
from util.losses import q_epsilon_greedy, q_td_error, q_huber_loss

from mlp.mlp import MLP
from kan.kan import KAN

class StreamingAgent:
    """Base class for streaming reinforcement learning agents."""
    def __init__(self, env_wrapper):
        self.env_wrapper = env_wrapper
        self.name = "BaseAgent"
    
    def select_action(self, state):
        """Select an action given the current state."""
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state, done):
        """Update the agent's knowledge based on experience."""
        raise NotImplementedError
        
    def run_episode(self, max_steps=1000, render=False):
        """Run a single episode."""
        self.key, key_reset = jr.split(self.key)
        obs, state = self.env_wrapper.reset(key_reset)
        
        episode_rewards = []
        episode_td_errors = []
        
        for step in range(max_steps):
            action_idx = self.select_action(obs)
            
            self.key, key_step = jr.split(self.key)
            env_action = self.env_wrapper.process_action(action_idx)
            
            next_obs, next_state, reward, done, info = self.env_wrapper.step(
                key_step, state, env_action
            )
            
            td_error = self.update(obs, action_idx, reward, next_obs, done)
            
            episode_rewards.append(reward)
            episode_td_errors.append(float(td_error))
            
            obs, state = next_obs, next_state
            
            if done:
                break
        
        self.episode_count += 1
        self.total_steps += step + 1
        self.total_rewards.append(sum(episode_rewards))
        self.avg_td_errors.append(sum(episode_td_errors) / (step + 1))
        
        return {
            'rewards': episode_rewards,
            'total_reward': sum(episode_rewards),
            'steps': step + 1,
            'avg_td_error': sum(episode_td_errors) / (step + 1)
        }
    
    def run(self, num_episodes=10, max_steps_per_episode=1000, verbose=True):
        """Run the agent for multiple episodes."""
        if verbose:
            episodes_iter = tqdm(range(num_episodes), desc=f"Training {self.name}")
        else:
            episodes_iter = range(num_episodes)
        
        episode_results = []
        
        for episode in episodes_iter:
            result = self.run_episode(max_steps=max_steps_per_episode)
            episode_results.append(result)

            if verbose:
                recent_rewards = [r['total_reward'] for r in episode_results[-10:]]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                recent_td_errors = [r['avg_td_error'] for r in episode_results[-10:]]
                avg_td_error = sum(recent_td_errors) / len(recent_td_errors)
                
                episodes_iter.set_description(
                    f"{self.name} | "
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {result['total_reward']:.2f} | "
                    f"10-Ep Avg: {avg_reward:.2f} | "
                    f"TD Error: {avg_td_error:.4f}"
                )
        
        return episode_results

class QTableWrapper:
    """Wrapper around Q-table to make it compatible with network interface."""
    def __init__(self, q_table, discretize_fn, num_actions):
        self.q_table = q_table
        self.discretize_fn = discretize_fn
        self._num_actions = num_actions
    
    def __call__(self, state):
        """Return Q-values for given state."""
        state_idx = self.discretize_fn(state)
        return self.q_table[state_idx]
    
    def num_actions(self):
        """Return number of actions, compatible with losses.py interface."""
        return self._num_actions

class TabularStreamQAgent(StreamingAgent):
    """
    Streaming Q-learning agent with tabular state-action value function.
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
        self.name = "TabularQAgent"
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
        
        self.q_wrapper = QTableWrapper(
            self.q_table, 
            self.discretize_state, 
            self.num_actions
        )
        
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = []
        self.avg_td_errors = []
    
    def discretize_state(self, state):
        """Convert continuous state to discrete indices for Q-table lookup."""
        indices = self.env_wrapper.discretize_state(state)
        return tuple(indices)
    
    def select_action(self, state):
        """Epsilon-greedy action selection using utility function."""
        self.key, subkey = jr.split(self.key)
        
        action, _, _ = q_epsilon_greedy(
            self.q_wrapper, 
            state, 
            self.epsilon, 
            subkey
        )
        
        return action
    
    def update(self, state, action_idx, reward, next_state, done):
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
        
        self.q_wrapper = QTableWrapper(
            self.q_table, 
            self.discretize_state, 
            self.num_actions
        )
        
        return td_error


class NetworkStreamQAgent(StreamingAgent):
    """
    Base class for streaming Q-learning agents.
    """
    def __init__(
        self,
        env_wrapper,
        network,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=0.1,
        seed=0
    ):
        super().__init__(env_wrapper)
        self.name = "NetworkQAgent"
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.key = jr.PRNGKey(seed)
        
        obs_space = env_wrapper.get_observation_space()
        act_space = env_wrapper.get_action_space()
        
        self.obs_dim = obs_space['dim']
        
        if act_space.get('is_discrete', False):
            self.action_is_discrete = True
            self.num_actions = act_space['num_bins']
        else:
            self.action_is_discrete = False
            self.discrete_actions = act_space['discrete_actions']
            self.num_actions = len(self.discrete_actions)
        
        self.network = network
        # ensure network has num_actions method for epsilon-greedy
        if not hasattr(self.network, 'num_actions'):
            self.network.num_actions = lambda: self.num_actions
        
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.network)
        
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = []
        self.avg_td_errors = []
    
    def select_action(self, state):
        """Epsilon-greedy action selection using utility function."""
        self.key, subkey = jr.split(self.key)
        
        state_array = jnp.array(state)
        action, _, _ = q_epsilon_greedy(
            self.network, 
            state_array, 
            self.epsilon, 
            subkey
        )
        
        return action
    
    def update(self, state, action_idx, reward, next_state, done):
        """Update Q-function approximator using TD learning."""
        state_array = jnp.array(state)
        next_state_array = jnp.array(next_state)
        
        td_error = q_td_error(
            self.network,
            state_array,
            action_idx,
            reward,
            done,
            next_state_array,
            self.gamma
        )
        
        def loss_fn(network):
            return q_huber_loss(
                network,
                state_array,
                action_idx,
                reward,
                done,
                next_state_array,
                self.gamma
            )
        
        grads = eqx.filter_grad(loss_fn)(self.network)
        
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.network
        )
        self.network = eqx.apply_updates(self.network, updates)
        
        return td_error

class MLPStreamQAgent(NetworkStreamQAgent):
    """
    Streaming Q-learning agent with MLP.
    """
    def __init__(
        self,
        env_wrapper,
        hidden_dims=[64, 32],
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=0.1,
        seed=0
    ):
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        
        obs_space = env_wrapper.get_observation_space()
        act_space = env_wrapper.get_action_space()
        
        obs_dim = obs_space['dim']
        
        if act_space.get('is_discrete', False):
            num_actions = act_space['num_bins']
        else:
            num_actions = len(act_space['discrete_actions'])
        
        network_dims = [obs_dim] + hidden_dims + [num_actions]
        network = MLP(network_dims, subkey)
        
        super().__init__(
            env_wrapper,
            network,
            learning_rate,
            discount_factor,
            epsilon,
            seed
        )
        
        self.name = "MLPQAgent"

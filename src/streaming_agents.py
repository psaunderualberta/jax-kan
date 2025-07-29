import jax
import optax
import equinox as eqx
from jax import random as jr, numpy as jnp, value_and_grad
from tqdm import tqdm
from util.losses import q_epsilon_greedy, q_td_error, q_huber_loss
from util import linear_epsilon_schedule
from mlp.mlp import MLP
from kan.kan import KAN
from gymnax import make


class StreamingAgent:
    """Base class for streaming reinforcement learning agents."""
    def __init__(self, env_name="CartPole-v1"):
        self.env, self.env_params = make(env_name)
        self.env_name = env_name
        self.name = "BaseAgent"
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = []
        self.avg_td_errors = []
    
    def select_action(self, state):
        """Select an action given the current state."""
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state, done):
        """Update the agent's knowledge based on experience."""
        raise NotImplementedError
        
    def run_episode(self, max_steps=1000, render=False):
        """Run a single episode."""
        self.key, key_reset = jr.split(self.key)
        obs, state = self.env.reset(key_reset, self.env_params)
        
        episode_rewards = []
        episode_td_errors = []
        
        for step in range(max_steps):
            action = self.select_action(obs)
            
            self.key, key_step = jr.split(self.key)
            next_obs, next_state, reward, done, info = self.env.step(
                key_step, state, action, self.env_params
            )

            self.key, key_step = jr.split(self.key)
            td_error = self.update(obs, action, reward, next_obs, done)
            
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
    
    def run(self, num_episodes=10, max_steps_per_episode=1000, verbose=True, save_plot=True, plot_path="training_plot.png"):
        """Run the agent for multiple episodes and save a plot of rewards and TD errors."""
        import matplotlib.pyplot as plt
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

        if save_plot:
            from util.plotting import plot_training_progress
            rewards = [r['total_reward'] for r in episode_results]
            td_errors = [r['avg_td_error'] for r in episode_results]
            plot_training_progress(rewards, td_errors, plot_path=f"{self.name}_{plot_path}", agent_name=self.name)

        return episode_results

class NetworkStreamQAgent(StreamingAgent):
    """
    Base class for streaming Q-learning agents.
    """
    def __init__(
        self,
        network,
        env_name="CartPole-v1",
        learning_rate=0.001,
        discount_factor=0.99,
        start_e=1.0,
        end_e=0.01,
        decay_duration=500,
        seed=0
    ):
        super().__init__(env_name)
        self.name = "NetworkQAgent"
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.start_e = start_e
        self.end_e = end_e
        self.decay_duration = decay_duration

        self.key = jr.PRNGKey(seed)
        
        obs_space = self.env.observation_space(self.env_params)
        self.obs_dim = len(obs_space.shape)
        self.num_actions = self.env.num_actions
        
        self.network = network
        
        # optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.network)
    
    def select_action(self, state):
        """Epsilon-greedy action selection using utility function."""
        self.key, subkey = jr.split(self.key)
        
        state_array = jnp.array(state)
        eps = linear_epsilon_schedule(
            self.start_e, 
            self.end_e, 
            self.decay_duration, 
            self.episode_count
        )
        action, _, _ = q_epsilon_greedy(
            self.network, 
            state_array, 
            eps,
            subkey
        )
        
        return action
    
    def update(self, state, action_idx, reward, next_state, done):
        """Update Q-function approximator using TD learning."""
        state_array = jnp.array(state).reshape(1, -1)
        next_state_array = jnp.array(next_state).reshape(1, -1)
        
        loss, grads = value_and_grad(q_huber_loss)(
            self.network,
            state_array,
            action_idx,
            reward,
            done,
            next_state_array,
            self.gamma
        )
        
        
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.network
        )
        self.network = eqx.apply_updates(self.network, updates)
        
        return loss

class MLPStreamQAgent(NetworkStreamQAgent):
    """
    Streaming Q-learning agent with MLP.
    """
    def __init__(
        self,
        env_name="CartPole-v1",
        hidden_dims=[64, 32],
        learning_rate=0.001,
        discount_factor=0.99,
        start_e=1.0,
        end_e=0.01,
        decay_duration=500,
        seed=0
    ):
        env, env_params = make(env_name)
        obs_space = env.observation_space(env_params)
        obs_dim = len(obs_space.shape)
        if obs_dim == 0:
            obs_dim = 1
        else:
            obs_dim = obs_space.shape[0]
        
        num_actions = env.num_actions
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [obs_dim] + hidden_dims + [num_actions]
        network = MLP(network_dims, subkey)
        
        super().__init__(
            network,
            env_name,
            learning_rate,
            discount_factor,
            start_e,
            end_e,
            decay_duration,
            seed
        )
        
        self.name = "MLPQAgent"


class KANStreamQAgent(NetworkStreamQAgent):
    """
    Streaming Q-learning agent with KAN.
    """
    def __init__(
        self,
        env_name="CartPole-v1",
        hidden_dims=[64, 32],
        learning_rate=0.001,
        discount_factor=0.99,
        start_e=1.0,
        end_e=0.01,
        decay_duration=500,
        grid=7,
        k=3, 
        num_stds=3,
        seed=0
    ):
        env, env_params = make(env_name)
        obs_space = env.observation_space(env_params)
        obs_dim = len(obs_space.shape)
        if obs_dim == 0:
            obs_dim = 1
        else:
            obs_dim = obs_space.shape[0]
        
        num_actions = env.num_actions
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [obs_dim] + hidden_dims + [num_actions]
        network = KAN(network_dims, grid, k, num_stds, subkey)
        
        super().__init__(
            network,
            env_name,
            learning_rate,
            discount_factor,
            start_e,
            end_e,
            decay_duration,
            seed
        )
        
        self.name = "KANQAgent"

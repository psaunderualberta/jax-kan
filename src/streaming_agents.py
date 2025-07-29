import jax
import optax
import equinox as eqx
from jax import random as jr, numpy as jnp, value_and_grad, tree as jt, lax as jax_lax
from tqdm import tqdm
from util.losses import q_epsilon_greedy, q_huber_loss, get_delta
from util import (
    linear_epsilon_schedule, 
    SampleMeanStats, 
    normalize_observation, 
    scale_reward,
    init_eligibility_trace,
    update_eligibility_trace,
    ObGD,
    is_none,
)
from mlp.mlp import MLP
from kan.kan import KAN
from gymnax import make
import chex
from gymnax.environments import environment


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

class MLPBasicStreaming(NetworkStreamQAgent):
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


class KANBasicStreaming(NetworkStreamQAgent):
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


class StreamQTrainState(eqx.Module):
    """Training state for StreamQ algorithm."""
    key: chex.PRNGKey
    done: bool
    obs: jnp.ndarray
    state: environment.EnvState
    z_w: eqx.Module
    q_network: eqx.Module
    reward_: float
    reward_trace: float
    global_timestep: int
    obs_stats: SampleMeanStats
    reward_stats: SampleMeanStats
    length: int

    def replace(self, **kwargs) -> 'StreamQTrainState':
        """Replace attributes in the training state with new values."""
        els = list(kwargs.items())
        return eqx.tree_at(
            lambda t: tuple(getattr(t, k) for k, _ in els),
            self,
            tuple(v for _, v in els),
            is_leaf=is_none,
        )


class BaseStreamQ(StreamingAgent):
    """
    StreamQ(Lambda) algorithm implementation for streaming RL.
    
    This implements the StreamQ algorithm with eligibility traces and online normalization.
    """
    def __init__(
        self,
        network,
        env_name="CartPole-v1",
        learning_rate=1.0,
        discount_factor=0.99,
        lambda_=0.8,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=500,
        seed=0
    ):
        super().__init__(env_name)
        self.name = "StreamQAgent"
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.lambda_ = lambda_
        self.kappa = kappa
        self.start_e = start_e
        self.end_e = end_e
        self.stop_exploring_timestep = stop_exploring_timestep

        self.key = jr.PRNGKey(seed)
        
        obs_space = self.env.observation_space(self.env_params)
        self.obs_dim = len(obs_space.shape)
        if self.obs_dim == 0:
            self.obs_dim = 1
        else:
            self.obs_dim = obs_space.shape[0]
        self.num_actions = self.env.num_actions
        
        self.network = network
        
        self._init_train_state()
    
    def _init_train_state(self):
        """Initialize the training state for StreamQ."""
        self.key, key_reset, key_ts = jr.split(self.key, 3)
        obs, state = self.env.reset(key_reset, self.env_params)
        
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)
        reward_stats = SampleMeanStats.new_params(())
        
        self.train_state = StreamQTrainState(
            key=key_ts,
            done=False,
            obs=obs,
            state=state,
            z_w=init_eligibility_trace(self.network),
            q_network=self.network,
            reward_=0.0,
            reward_trace=0.0,
            global_timestep=0,
            length=0,
            obs_stats=obs_stats,
            reward_stats=reward_stats,
        )
    
    def select_action(self, state):
        """Epsilon-greedy action selection using StreamQ epsilon schedule."""
        self.key, subkey = jr.split(self.key)
        
        eps = linear_epsilon_schedule(
            self.start_e, 
            self.end_e, 
            self.stop_exploring_timestep, 
            self.train_state.global_timestep
        )
        action, _, _ = q_epsilon_greedy(
            self.train_state.q_network, 
            state, 
            eps,
            subkey
        )
        
        return action
    
    def update(self, state, action_idx, reward, next_state, done):
        """Update using StreamQ(Lambda) algorithm."""
        # normalize observations and reward
        next_state, new_obs_stats = normalize_observation(next_state, self.train_state.obs_stats)
        scaled_reward, new_reward_trace, new_reward_stats = scale_reward(
            reward, self.train_state.reward_stats, self.train_state.reward_trace, done, self.gamma
        )
        
        # compute TD error and gradients
        td_error, td_grad = value_and_grad(get_delta)(
            self.train_state.q_network, 
            scaled_reward, 
            self.gamma, 
            done, 
            state, 
            action_idx, 
            next_state
        )
        
        # update eligibility trace
        new_z_w = update_eligibility_trace(
            self.train_state.z_w, 
            self.gamma, 
            self.lambda_, 
            td_grad
        )
        
        # update Q-network using ObGD
        new_q_network = ObGD(
            new_z_w, 
            self.train_state.q_network, 
            td_error, 
            self.alpha, 
            self.kappa
        )
        
        # check if exploration occurred for trace reset
        self.key, action_key = jr.split(self.key)
        eps = linear_epsilon_schedule(
            self.start_e, 
            self.end_e, 
            self.stop_exploring_timestep, 
            self.train_state.global_timestep
        )
        _, _, explored = q_epsilon_greedy(
            self.train_state.q_network, 
            state, 
            eps, 
            action_key
        )
        
        # reset eligibility trace if exploration occurred or episode ended
        reset_condition = jnp.logical_or(explored, done)
        new_z_w = jt.map(
            lambda old: jax_lax.select(
                reset_condition,
                jnp.zeros_like(old),
                old
            ), 
            new_z_w
        )
        
        # update training state
        self.train_state = self.train_state.replace(
            obs=next_state,
            z_w=new_z_w,
            q_network=new_q_network,
            reward_=self.train_state.reward_ * self.gamma + reward,
            reward_trace=new_reward_trace,
            global_timestep=self.train_state.global_timestep + 1,
            length=self.train_state.length + 1,
            obs_stats=new_obs_stats,
            reward_stats=new_reward_stats,
        )
        
        # reset reward accumulation if episode ended
        if done:
            self.train_state = self.train_state.replace(
                reward_trace=0.0,
                reward_=0.0,
                length=0
            )
        
        # update network reference
        self.network = new_q_network
        
        return float(td_error)


class MLPStreamQLambda(BaseStreamQ):
    """
    StreamQ(Lambda) agent with MLP network.
    """
    def __init__(
        self,
        env_name="CartPole-v1",
        hidden_dims=[64, 32],
        learning_rate=1.0,
        discount_factor=0.99,
        lambda_=0.8,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=500,
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
            lambda_,
            kappa,
            start_e,
            end_e,
            stop_exploring_timestep,
            seed
        )
        
        self.name = "MLPStreamQLambda"


class KANStreamQLambda(BaseStreamQ):
    """
    StreamQ(Lambda) agent with KAN network.
    """
    def __init__(
        self,
        env_name="CartPole-v1",
        hidden_dims=[64, 32],
        learning_rate=1.0,
        discount_factor=0.99,
        lambda_=0.8,
        kappa=2.0,
        start_e=1.0,
        end_e=0.01,
        stop_exploring_timestep=500,
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
            lambda_,
            kappa,
            start_e,
            end_e,
            stop_exploring_timestep,
            seed
        )
        
        self.name = "KANStreamQLambda"

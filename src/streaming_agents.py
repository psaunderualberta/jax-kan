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
from collections import deque


from collections import deque


class ActionHistoryBuffer:
    """Buffer to store the last N state-action pairs for context."""
    def __init__(self, max_history=4, obs_dim=4, num_actions=2):
        self.max_history = max_history
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.history = deque(maxlen=max_history)
        self.reset()
    
    def reset(self):
        """Reset the history buffer."""
        self.history.clear()
        for _ in range(self.max_history):
            self.history.append({
                'state': jnp.zeros(self.obs_dim),
                'action': 0,
                'action_onehot': jnp.zeros(self.num_actions)
            })
    
    def add(self, state, action):
        """Add a new state-action pair to the history."""
        action_onehot = jnp.zeros(self.num_actions)
        action_onehot = action_onehot.at[action].set(1.0)
        
        self.history.append({
            'state': state,
            'action': action,
            'action_onehot': action_onehot
        })
    
    def get_extended_state(self, current_state):
        """Get the extended state representation including action history."""
        extended_state = [current_state]
        for entry in self.history:
            extended_state.append(entry['state'])
            extended_state.append(entry['action_onehot'])
        
        return jnp.concatenate(extended_state)
    
    def get_extended_dim(self):
        """Get the dimension of the extended state representation."""
        return self.obs_dim + self.max_history * (self.obs_dim + self.num_actions)


class StreamingAgent:
    """Base class for streaming reinforcement learning agents."""
    def __init__(self, env_name="CartPole-v1", use_action_history=True, history_length=4):
        self.env, self.env_params = make(env_name)
        self.env_name = env_name
        self.name = "BaseAgent"
        self.episode_count = 0
        self.total_steps = 0
        self.total_rewards = []
        self.avg_td_errors = []
        
        self.use_action_history = use_action_history
        self.history_length = history_length
        
        if self.use_action_history:
            obs_space = self.env.observation_space(self.env_params)
            obs_dim = len(obs_space.shape)
            if obs_dim == 0:
                obs_dim = 1
            else:
                obs_dim = obs_space.shape[0]
            
            self.action_history = ActionHistoryBuffer(
                max_history=history_length,
                obs_dim=obs_dim,
                num_actions=self.env.num_actions
            )
    
    def reset_episode(self):
        """Reset episode-specific state including action history."""
        if self.use_action_history:
            self.action_history.reset()
    
    def get_state_representation(self, state):
        """Get state representation (with or without action history)."""
        if self.use_action_history:
            return self.action_history.get_extended_state(state)
        else:
            return state
    
    def update_action_history(self, state, action):
        """Update the action history buffer."""
        if self.use_action_history:
            self.action_history.add(state, action)
    
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
        
        self.reset_episode()
        
        episode_rewards = []
        episode_td_errors = []
        
        for step in range(max_steps):
            state_repr = self.get_state_representation(obs)
            action = self.select_action(state_repr)
            
            self.key, key_step = jr.split(self.key)
            next_obs, next_state, reward, done, info = self.env.step(
                key_step, state, action, self.env_params
            )

            self.update_action_history(obs, action)
            
            next_state_repr = self.get_state_representation(next_obs)
            
            self.key, key_step = jr.split(self.key)
            td_error = self.update(state_repr, action, reward, next_state_repr, done)
            
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
        use_action_history=True,
        history_length=4,
        seed=0
    ):
        super().__init__(env_name, use_action_history, history_length)
        self.name = "NetworkQAgent"
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.start_e = start_e
        self.end_e = end_e
        self.decay_duration = decay_duration

        self.key = jr.PRNGKey(seed)
        
        obs_space = self.env.observation_space(self.env_params)
        self.obs_dim = len(obs_space.shape)
        if self.obs_dim == 0:
            self.obs_dim = 1
        else:
            self.obs_dim = obs_space.shape[0]
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
        use_action_history=True,
        history_length=4,
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
        
        if use_action_history:
            input_dim = obs_dim + history_length * (obs_dim + num_actions)
        else:
            input_dim = obs_dim
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [input_dim] + hidden_dims + [num_actions]
        network = MLP(network_dims, subkey)
        
        super().__init__(
            network,
            env_name,
            learning_rate,
            discount_factor,
            start_e,
            end_e,
            decay_duration,
            use_action_history,
            history_length,
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
        use_action_history=True,
        history_length=4,
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
        
        if use_action_history:
            # current state + history_length * (state + action_onehot)
            input_dim = obs_dim + history_length * (obs_dim + num_actions)
        else:
            input_dim = obs_dim
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [input_dim] + hidden_dims + [num_actions]
        network = KAN(network_dims, grid, k, num_stds, subkey)
        
        super().__init__(
            network,
            env_name,
            learning_rate,
            discount_factor,
            start_e,
            end_e,
            decay_duration,
            use_action_history,
            history_length,
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
    history_states: jnp.ndarray   # (history_length, obs_dim)
    history_actions: jnp.ndarray  # (history_length,)
    history_onehot: jnp.ndarray   # (history_length, num_actions)

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
        use_action_history=True,
        history_length=4,
        seed=0
    ):
        super().__init__(env_name, use_action_history, history_length)
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
    
    def get_extended_state_jax(self, current_state, history_states, history_actions, history_onehot):
        """JAX-compatible version of getting extended state representation."""
        if self.use_action_history:
            flat_history_states = history_states.reshape(-1)  # (history_length * obs_dim,)
            flat_history_onehot = history_onehot.reshape(-1)  # (history_length * num_actions,)
            
            return jnp.concatenate([current_state, flat_history_states, flat_history_onehot])
        else:
            return current_state
    
    def update_history_jax(self, history_states, history_actions, history_onehot, new_state, new_action):
        """JAX-compatible version of updating action history."""
        # shift existing history by one position (remove oldest, add newest)
        new_history_states = jnp.roll(history_states, -1, axis=0)
        new_history_states = new_history_states.at[-1].set(new_state)
        
        new_history_actions = jnp.roll(history_actions, -1)
        new_history_actions = new_history_actions.at[-1].set(new_action)
        
        new_action_onehot = jnp.zeros(self.num_actions)
        new_action_onehot = new_action_onehot.at[new_action].set(1.0)
        
        new_history_onehot = jnp.roll(history_onehot, -1, axis=0)
        new_history_onehot = new_history_onehot.at[-1].set(new_action_onehot)
        
        return new_history_states, new_history_actions, new_history_onehot
    
    def _init_train_state(self):
        """Initialize the training state for StreamQ."""
        self.key, key_reset, key_ts = jr.split(self.key, 3)
        obs, state = self.env.reset(key_reset, self.env_params)
        
        obs_stats = SampleMeanStats.new_params(obs.shape)
        obs, obs_stats = normalize_observation(obs, obs_stats)
        reward_stats = SampleMeanStats.new_params(())
        
        history_states = jnp.zeros((self.history_length, self.obs_dim))
        history_actions = jnp.zeros(self.history_length, dtype=jnp.int32)
        history_onehot = jnp.zeros((self.history_length, self.num_actions))
        
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
            history_states=history_states,
            history_actions=history_actions,
            history_onehot=history_onehot,
        )
    
    def select_action(self, state):
        """Epsilon-greedy action selection using StreamQ epsilon schedule."""
        self.key, subkey = jr.split(self.key)
        
        if self.use_action_history:
            extended_state = self.get_extended_state_jax(
                state, 
                self.train_state.history_states,
                self.train_state.history_actions,
                self.train_state.history_onehot
            )
        else:
            extended_state = state
        
        eps = linear_epsilon_schedule(
            self.start_e, 
            self.end_e, 
            self.stop_exploring_timestep, 
            self.train_state.global_timestep
        )
        action, _, _ = q_epsilon_greedy(
            self.train_state.q_network, 
            extended_state, 
            eps,
            subkey
        )
        
        return action
    
    def update(self, state, action_idx, reward, next_state, done):
        """Update using StreamQ(Lambda) algorithm."""
        # Note: state and next_state are already extended representations from run_episode
        
        # normalize observations and reward
        # for normalization, we use only the raw observation part, not the extended state
        if self.use_action_history:
            # extract raw observation from extended state (first obs_dim elements)
            raw_next_obs = next_state[:self.obs_dim]
        else:
            raw_next_obs = next_state
            
        raw_next_obs, new_obs_stats = normalize_observation(raw_next_obs, self.train_state.obs_stats)
        
        # update the extended state representation with normalized observation
        if self.use_action_history:
            next_state_normalized = jnp.concatenate([
                raw_next_obs,
                next_state[self.obs_dim:]
            ])
        else:
            next_state_normalized = raw_next_obs
        
        scaled_reward, new_reward_trace, new_reward_stats = scale_reward(
            reward, self.train_state.reward_stats, self.train_state.reward_trace, done, self.gamma
        )
        
        # compute TD error and gradients using extended states
        td_error, td_grad = value_and_grad(get_delta)(
            self.train_state.q_network, 
            scaled_reward, 
            self.gamma, 
            done, 
            state,
            action_idx, 
            next_state_normalized
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
            state,  # use current extended state for exploration check
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
        
        if self.use_action_history:
            raw_current_obs = state[:self.obs_dim]
            new_history_states, new_history_actions, new_history_onehot = self.update_history_jax(
                self.train_state.history_states,
                self.train_state.history_actions, 
                self.train_state.history_onehot,
                raw_current_obs,
                action_idx
            )
        else:
            new_history_states = self.train_state.history_states
            new_history_actions = self.train_state.history_actions
            new_history_onehot = self.train_state.history_onehot
        
        # update training state
        self.train_state = self.train_state.replace(
            obs=raw_next_obs,
            z_w=new_z_w,
            q_network=new_q_network,
            reward_=self.train_state.reward_ * self.gamma + reward,
            reward_trace=new_reward_trace,
            global_timestep=self.train_state.global_timestep + 1,
            length=self.train_state.length + 1,
            obs_stats=new_obs_stats,
            reward_stats=new_reward_stats,
            history_states=new_history_states,
            history_actions=new_history_actions,
            history_onehot=new_history_onehot,
        )
        
        # reset reward accumulation and history if episode ended
        if done:
            reset_history_states = jnp.zeros((self.history_length, self.obs_dim))
            reset_history_actions = jnp.zeros(self.history_length, dtype=jnp.int32)
            reset_history_onehot = jnp.zeros((self.history_length, self.num_actions))
            
            self.train_state = self.train_state.replace(
                reward_trace=0.0,
                reward_=0.0,
                length=0,
                history_states=reset_history_states,
                history_actions=reset_history_actions,
                history_onehot=reset_history_onehot,
            )
        
        # update network reference
        self.network = new_q_network
        
        return float(td_error)
    
    def run_episode(self, max_steps=1000, render=False):
        """Run a single episode with StreamQ-specific state management."""
        self.key, key_reset = jr.split(self.key)
        obs, state = self.env.reset(key_reset, self.env_params)
        
        self._init_train_state()
        
        episode_rewards = []
        episode_td_errors = []
        
        for step in range(max_steps):
            action = self.select_action(self.train_state.obs)
            
            self.key, key_step = jr.split(self.key)
            next_obs, next_state, reward, done, info = self.env.step(
                key_step, state, action, self.env_params
            )

            if self.use_action_history:
                current_extended_state = self.get_extended_state_jax(
                    self.train_state.obs, 
                    self.train_state.history_states,
                    self.train_state.history_actions,
                    self.train_state.history_onehot
                )
                temp_history_states, temp_history_actions, temp_history_onehot = self.update_history_jax(
                    self.train_state.history_states,
                    self.train_state.history_actions, 
                    self.train_state.history_onehot,
                    self.train_state.obs,
                    action
                )
                next_extended_state = self.get_extended_state_jax(
                    next_obs,
                    temp_history_states,
                    temp_history_actions,
                    temp_history_onehot
                )
            else:
                current_extended_state = self.train_state.obs
                next_extended_state = next_obs
            
            self.key, key_step = jr.split(self.key)
            td_error = self.update(current_extended_state, action, reward, next_extended_state, done)
            
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
        use_action_history=True,
        history_length=4,
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
        
        if use_action_history:
            # current state + history_length * (state + action_onehot)
            input_dim = obs_dim + history_length * (obs_dim + num_actions)
        else:
            input_dim = obs_dim
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [input_dim] + hidden_dims + [num_actions]
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
            use_action_history,
            history_length,
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
        use_action_history=True,
        history_length=4,
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
        
        if use_action_history:
            # current state + history_length * (state + action_onehot)
            input_dim = obs_dim + history_length * (obs_dim + num_actions)
        else:
            input_dim = obs_dim
        
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)
        network_dims = [input_dim] + hidden_dims + [num_actions]
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
            use_action_history,
            history_length,
            seed
        )
        
        self.name = "KANStreamQLambda"

"""
Weights & Biases hyperparameter sweeps for StreamQ algorithms.
"""

import argparse
import jax
import wandb
import numpy as np
from jax import random as jr, numpy as jnp
from streaming_agents import (
    MLPBasicStreaming, 
    KANBasicStreaming, 
    MLPStreamQLambda, 
    KANStreamQLambda
)


BASIC_STREAMING_SWEEP_CONFIG = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward','goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values','min': 1e-5,'max': 1e-2},
        'discount_factor': {'values': [0.99]},
        'start_e': {'value': 1.0},
        'end_e': {'min': 0.01,'max': 0.2},
        'decay_duration': {'values': [500, 1000, 2000, 5000]},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'distribution': 'int_uniform','min': 3,'max': 12},
        'kan_k': {'distribution': 'int_uniform', 'min': 2,'max': 6},
        'kan_num_stds': {'distribution': 'int_uniform','min': 1,'max': 6}
    }
}

STREAMQ_LAMBDA_SWEEP_CONFIG = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward','goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values','min': 1e-5,'max': 1e-2},
        'discount_factor': {'values': [0.99]},
        'lambda_': {'distribution': 'uniform','min': 0.0,'max': 1.0},
        'kappa': {'distribution': 'log_uniform_values','min': 0.1,'max': 10.0},
        'start_e': {'value': 1.0},
        'end_e': {'min': 0.01,'max': 0.2},
        'exploration_fraction': {'min': 0.1,'max': 0.8},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'distribution': 'int_uniform','min': 3,'max': 12},
        'kan_k': {'distribution': 'int_uniform', 'min': 2,'max': 6},
        'kan_num_stds': {'distribution': 'int_uniform','min': 1,'max': 6}
    }
}


def train_and_evaluate_agent(agent, num_episodes, max_steps, log_frequency=1, algorithm=None):
    """Train an agent and log metrics to wandb."""
    
    print(f"Starting training for {algorithm} agent...")
    results = agent.run(
        num_episodes=num_episodes, 
        max_steps_per_episode=max_steps,
        verbose=False,
        save_plot=False
    )
    
    total_rewards = agent.total_rewards
    episode_lengths = [r['steps'] for r in results]
    td_errors = [r['avg_td_error'] for r in results]
    
    total_steps = np.sum(episode_lengths)
    timesteps_per_episode = np.cumsum(episode_lengths)
    
    steps_x = timesteps_per_episode
    rewards_y = total_rewards
    
    window_size = 20
    
    print(f"Logging metrics to W&B for {len(total_rewards)} episodes...")
    for i, (reward, length, td_error) in enumerate(zip(total_rewards, episode_lengths, td_errors)):
        if i % log_frequency == 0 or i == len(total_rewards) - 1:
            if i >= window_size:
                reward_window = total_rewards[i-window_size:i]
                length_window = episode_lengths[i-window_size:i]
                td_window = td_errors[i-window_size:i]
                
                reward_mean = np.mean(reward_window)
                reward_var = np.var(reward_window)
                length_mean = np.mean(length_window)
                length_var = np.var(length_window)
                td_mean = np.mean(td_window)
            else:
                reward_mean = reward
                reward_var = 0.0
                length_mean = length
                length_var = 0.0
                td_mean = td_error
            
            if i > 0:
                current_step = timesteps_per_episode[i]
                timestep_window = 1000
                window_episodes = [
                    j for j, step in enumerate(timesteps_per_episode[:i+1]) 
                    if step > current_step - timestep_window
                ]
                timestep_avg_reward = np.mean([total_rewards[j] for j in window_episodes])
            else:
                timestep_avg_reward = reward
                
            if i % 10 == 0 or i == len(total_rewards) - 1:
                print(f"Episode {i}/{num_episodes} | Reward: {reward:.2f} | " +
                      f"Avg(20): {reward_mean:.2f} | Steps: {timesteps_per_episode[i]} | " +
                      f"Progress: {(i / num_episodes) * 100:.1f}%")
                
            metrics = {
                # Episode counters
                'episode': i,
                'timesteps': timesteps_per_episode[i],
                
                # Raw episode stats
                'reward': reward,
                'length': length,
                'td_error': td_error,
                
                # Window statistics (episode-based)
                'reward_mean': reward_mean,
                'reward_var': reward_var, 
                'length_mean': length_mean,
                'length_var': length_var,
                'td_error_mean': td_mean,
                
                # Timestep-based statistics
                'timestep_avg_reward': timestep_avg_reward,
                
                # Progress tracking
                'percent_complete': (i / num_episodes) * 100
            }
            wandb.log(metrics)
    
    final_avg_reward = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
    final_avg_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
    final_avg_td_error = np.mean(td_errors[-10:]) if len(td_errors) >= 10 else np.mean(td_errors)
    max_reward = np.max(total_rewards)
    stability = np.std(total_rewards[-10:]) if len(total_rewards) >= 10 else np.std(total_rewards)
    
    summary_metrics = {
        'final_avg_reward': final_avg_reward,
        'final_avg_length': final_avg_length,
        'final_avg_td_error': final_avg_td_error,
        'max_reward': max_reward,
        'reward_stability': stability,
        'total_timesteps': total_steps,
        'episodes_completed': len(total_rewards)
    }
    
    table_data = []
    for i, (reward, length, td_error) in enumerate(zip(total_rewards, episode_lengths, td_errors)):
        table_data.append([
            i,
            timesteps_per_episode[i],
            reward,
            length,
            td_error
        ])
    
    columns = ["episode", "timesteps", "reward", "length", "td_error"]
    episodes_table = wandb.Table(data=table_data, columns=columns)
    
    print("\nTraining completed!")
    print(f"Final average reward (last 10 episodes): {final_avg_reward:.2f}")
    print(f"Max reward achieved: {max_reward:.2f}")
    print(f"Reward stability (std): {stability:.4f}")
    print(f"Total timesteps: {total_steps}")
    
    wandb.log(summary_metrics)
    wandb.log({"episodes_data": episodes_table})
    
    # Filter out NaN values before creating histograms
    valid_rewards = [r for r in total_rewards if not np.isnan(r)]
    valid_lengths = [l for l in episode_lengths if not np.isnan(l)]
    valid_td_errors = [td for td in td_errors if not np.isnan(td) and np.isfinite(td)]
    
    histograms = {}
    if valid_rewards:
        histograms["reward_histogram"] = wandb.Histogram(valid_rewards)
    if valid_lengths:
        histograms["length_histogram"] = wandb.Histogram(valid_lengths)
    if valid_td_errors:
        histograms["td_error_histogram"] = wandb.Histogram(valid_td_errors)
    
    if histograms:
        wandb.log(histograms)
    
    return final_avg_reward, results


def train_basic_streaming_mlp():
    """Training function for basic streaming MLP agent."""
    config = wandb.config
    
    agent = MLPBasicStreaming(
        env_name=config.env_name,
        hidden_dims=config.hidden_dims,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        start_e=config.start_e,
        end_e=config.end_e,
        decay_duration=config.decay_duration,
        seed=config.seed
    )
    
    final_reward, _ = train_and_evaluate_agent(
        agent, config.num_episodes, config.max_steps, algorithm='basic'
    )
    
    return final_reward


def train_basic_streaming_kan():
    """Training function for basic streaming KAN agent."""
    config = wandb.config
    
    agent = KANBasicStreaming(
        env_name=config.env_name,
        hidden_dims=config.hidden_dims,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        start_e=config.start_e,
        end_e=config.end_e,
        decay_duration=config.decay_duration,
        grid=config.kan_grid,
        k=config.kan_k,
        num_stds=config.kan_num_stds,
        seed=config.seed
    )
    
    final_reward, _ = train_and_evaluate_agent(
        agent, config.num_episodes, config.max_steps, algorithm='basic'
    )
    
    return final_reward


def train_streamq_mlp():
    """Training function for StreamQ MLP agent."""
    config = wandb.config
    
    total_timesteps = config.num_episodes * config.max_steps
    stop_exploring_timestep = int(total_timesteps * config.exploration_fraction)
    
    agent = MLPStreamQLambda(
        env_name=config.env_name,
        hidden_dims=config.hidden_dims,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        lambda_=config.lambda_,
        kappa=config.kappa,
        start_e=config.start_e,
        end_e=config.end_e,
        stop_exploring_timestep=stop_exploring_timestep,
        seed=config.seed
    )
    
    final_reward, _ = train_and_evaluate_agent(
        agent, config.num_episodes, config.max_steps, algorithm='streamq'
    )
    
    return final_reward


def train_streamq_kan():
    """Training function for StreamQ KAN agent."""
    config = wandb.config
    
    total_timesteps = config.num_episodes * config.max_steps
    stop_exploring_timestep = int(total_timesteps * config.exploration_fraction)
    
    agent = KANStreamQLambda(
        env_name=config.env_name,
        hidden_dims=config.hidden_dims,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        lambda_=config.lambda_,
        kappa=config.kappa,
        start_e=config.start_e,
        end_e=config.end_e,
        stop_exploring_timestep=stop_exploring_timestep,
        grid=config.kan_grid,
        k=config.kan_k,
        num_stds=config.kan_num_stds,
        seed=config.seed
    )
    
    final_reward, _ = train_and_evaluate_agent(
        agent, config.num_episodes, config.max_steps, algorithm='streamq'
    )
    
    return final_reward


def run_sweep_agent():
    """Main function to run a single sweep trial."""
    with wandb.init() as run:
        config = wandb.config
        
        # jax.config.update('jax_platform_name', 'cpu')
        
        if config.algorithm == 'basic' and config.network == 'mlp':
            final_reward = train_basic_streaming_mlp()
        elif config.algorithm == 'basic' and config.network == 'kan':
            final_reward = train_basic_streaming_kan()
        elif config.algorithm == 'streamq' and config.network == 'mlp':
            final_reward = train_streamq_mlp()
        elif config.algorithm == 'streamq' and config.network == 'kan':
            final_reward = train_streamq_kan()
        else:
            raise ValueError(f"Unknown combination: {config.algorithm} + {config.network}")
        
        return final_reward


def create_sweep(algorithm, network, env_name="CartPole-v1", project_name="streamq-hyperopt"):
    """Create a wandb sweep for the specified algorithm and network."""
    
    if algorithm == 'basic':
        sweep_config = BASIC_STREAMING_SWEEP_CONFIG.copy()
    elif algorithm == 'streamq':
        sweep_config = STREAMQ_LAMBDA_SWEEP_CONFIG.copy()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Add program specification for wandb agent
    # Use the full command specification to ensure correct python interpreter
    sweep_config['command'] = [
        '/home/ammany01/.pyenv/versions/env-3.11.13/bin/python',
        'src/wandb_hyperparam_sweep.py',
        '${args}'
    ]
    
    sweep_config['parameters'].update({
        'algorithm': {'value': algorithm},
        'network': {'value': network},
        'env_name': {'value': env_name},
        'num_episodes': {'value': 200},
        'max_steps': {'value': 500},
        'seed': {'value': 0}
    })
    
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name,
        entity=None
    )
    
    print(f"Created sweep {sweep_id} for {algorithm}-{network}")
    print(f"Run with: wandb agent {sweep_id}")
    
    return sweep_id


def run_single_experiment(algorithm, network, config_override=None):
    """Run a single experiment with specified configuration."""
    
    if algorithm == 'basic':
        default_config = {
            'algorithm': algorithm,
            'network': network,
            'env_name': 'CartPole-v1',
            # 'num_episodes': 200,
            'num_episodes': 10,
            # 'max_steps': 500,
            'max_steps': 100,
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'start_e': 1.0,
            'end_e': 0.01,
            'decay_duration': 500,
            'hidden_dims': (64, 32),
            'kan_grid': 7,
            'kan_k': 3,
            'kan_num_stds': 3,
            'seed': 0
        }
    else:
        default_config = {
            'algorithm': algorithm,
            'network': network,
            'env_name': 'CartPole-v1',
            # 'num_episodes': 200,
            'num_episodes': 10,
            # 'max_steps': 500,
            'max_steps': 100,
            'learning_rate': 1.0,
            'discount_factor': 0.99,
            'lambda_': 0.8,
            'kappa': 2.0,
            'start_e': 1.0,
            'end_e': 0.01,
            'exploration_fraction': 0.5,
            'hidden_dims': (64, 32),
            'kan_grid': 7,
            'kan_k': 3,
            'kan_num_stds': 3,
            'seed': 0
        }
    
    if config_override:
        default_config.update(config_override)
    
    with wandb.init(
        project="streamq-hyperopt",
        config=default_config,
        name=f"{algorithm}-{network}-single"
    ) as run:
        final_reward = run_sweep_agent()
        print(f"Final reward: {final_reward:.2f}")
        return final_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter sweeps with Weights & Biases')
    parser.add_argument('--mode', choices=['sweep', 'agent', 'single'], 
                       help='Mode: create sweep, run sweep agent, or run single experiment')
    parser.add_argument('--algorithm', choices=['basic', 'streamq'], 
                       help='Algorithm to optimize')
    parser.add_argument('--network', choices=['mlp', 'kan'], 
                       help='Network type')
    parser.add_argument('--env', default='CartPole-v1', help='Environment name')
    parser.add_argument('--project', default='streamq-hyperopt', help='Wandb project name')
    parser.add_argument('--sweep_id', help='Sweep ID for agent mode (optional when called by wandb agent)')
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs')
    
    # Additional parameters that wandb agent might pass
    parser.add_argument('--discount_factor', type=float, help='Discount factor')
    parser.add_argument('--decay_duration', type=int, help='Epsilon decay duration')
    parser.add_argument('--end_e', type=float, help='End epsilon')
    parser.add_argument('--env_name', help='Environment name')
    parser.add_argument('--exploration_fraction', type=float, help='Exploration fraction')
    parser.add_argument('--hidden_dims', help='Hidden dimensions')
    parser.add_argument('--kan_grid', type=int, help='KAN grid size')
    parser.add_argument('--kan_k', type=int, help='KAN k parameter')
    parser.add_argument('--kan_num_stds', type=int, help='KAN num_stds parameter')
    parser.add_argument('--kappa', type=float, help='Kappa parameter')
    parser.add_argument('--lambda_', type=float, help='Lambda parameter')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_steps', type=int, help='Max steps per episode')
    parser.add_argument('--num_episodes', type=int, help='Number of episodes')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--start_e', type=float, help='Start epsilon')
    
    args = parser.parse_args()
    
    # If wandb agent calls this script, it will pass hyperparameters directly
    # Check if we have hyperparameters but no mode specified
    if not args.mode and args.algorithm and args.network:
        # This means we're being called by wandb agent
        print(f"Running sweep agent for {args.algorithm}-{args.network}")
        run_sweep_agent()
    elif args.mode == 'sweep':
        if not args.algorithm or not args.network:
            print("Error: --algorithm and --network required for sweep mode")
            exit(1)
        sweep_id = create_sweep(
            args.algorithm, 
            args.network, 
            args.env, 
            args.project
        )
        print(f"\nTo run the sweep, execute:")
        print(f"python {__file__} --mode agent --algorithm {args.algorithm} --network {args.network} --sweep_id {sweep_id}")
        
    elif args.mode == 'agent':
        # When called by wandb agent, sweep_id might not be provided
        if not args.algorithm or not args.network:
            print("Error: --algorithm and --network required for agent mode")
            exit(1)
        print(f"Running sweep agent for {args.algorithm}-{args.network}")
        if args.sweep_id:
            wandb.agent(args.sweep_id, run_sweep_agent, count=args.count)
        else:
            # If no sweep_id provided, just run the agent function directly
            # This happens when wandb agent calls the script with parameters
            run_sweep_agent()
        
    elif args.mode == 'single':
        if not args.algorithm or not args.network:
            print("Error: --algorithm and --network required for single mode")
            exit(1)
        print(f"Running single experiment for {args.algorithm}-{args.network}")
        final_reward = run_single_experiment(args.algorithm, args.network)
        print(f"Experiment completed with final reward: {final_reward:.2f}")
    else:
        print("Error: Mode required (or script called by wandb agent)")
        parser.print_help()
        exit(1)

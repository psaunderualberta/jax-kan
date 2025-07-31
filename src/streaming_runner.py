import jax
import argparse
import sys
from jax import random as jr, numpy as jnp, lax

from streaming_agents import (
    MLPBasicStreaming, 
    KANBasicStreaming, 
    MLPStreamQLambda, 
    KANStreamQLambda
)

def test_basic_streaming_agents(env_name, num_episodes=100, max_steps=500, network='both'):
    """Test basic streaming Q-learning agents (MLP and KAN)."""
    print("="*60)
    print("TESTING BASIC STREAMING Q-LEARNING AGENTS")
    print("="*60)
    
    mlp_agent, kan_agent = None, None
    mlp_results, kan_results = None, None

    if network in ['mlp', 'both']:
        print("\nTraining MLP Basic Streaming Q-Learning Agent:")
        mlp_agent = MLPBasicStreaming(
            env_name=env_name,
            hidden_dims=[64, 32],
            learning_rate=0.001,
            discount_factor=0.99,
            start_e=1.0,
            end_e=0.01,
            decay_duration=500,
            use_action_history=True,
            history_length=4,
            seed=0
        )
        mlp_results = mlp_agent.run(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        print(f"MLP Agent - Final Average Reward: {sum(mlp_agent.total_rewards[-3:]) / 3:.2f}")

    if network in ['kan', 'both']:
        print("\nTraining KAN Basic Streaming Q-Learning Agent:")
        kan_agent = KANBasicStreaming(
            env_name=env_name,
            hidden_dims=[128],
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
        )
        kan_results = kan_agent.run(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        print(f"KAN Agent - Final Average Reward: {sum(kan_agent.total_rewards[-3:]) / 3:.2f}")

    
    return mlp_agent, kan_agent, mlp_results, kan_results


def test_streamq_lambda_agents(env_name, num_episodes=100, max_steps=500, network='both'):
    """Test StreamQ(Lambda) agents (MLP and KAN)."""
    print("="*60)
    print("TESTING STREAMQ(LAMBDA) AGENTS")
    print("="*60)
    
    mlp_streamq_agent, kan_streamq_agent = None, None
    mlp_streamq_results, kan_streamq_results = None, None

    if network in ['mlp', 'both']:
        print("\nTraining MLP StreamQ(Lambda) Agent:")
        mlp_streamq_agent = MLPStreamQLambda(
            env_name=env_name,
            hidden_dims=[64, 32],
            learning_rate=1.0,
            discount_factor=0.99,
            lambda_=0.8,
            kappa=2.0,
            start_e=1.0,
            end_e=0.01,
            stop_exploring_timestep=int(num_episodes * max_steps * 0.5),
            use_action_history=True,
            history_length=4,
            seed=0
        )
        mlp_streamq_results = mlp_streamq_agent.run(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        print(f"MLP StreamQ Agent - Final Average Reward: {sum(mlp_streamq_agent.total_rewards[-3:]) / 3:.2f}")

    
    if network in ['kan', 'both']:
        print("\nTraining KAN StreamQ(Lambda) Agent:")
        kan_streamq_agent = KANStreamQLambda(
            env_name=env_name,
            hidden_dims=[128],
            learning_rate=1.0,
            discount_factor=0.99,
            lambda_=0.8,
            kappa=2.0,
            start_e=1.0,
            end_e=0.01,
            stop_exploring_timestep=int(num_episodes * max_steps * 0.5),
            grid=7,
            k=3,
            num_stds=3,
            use_action_history=True,
            history_length=4,
            seed=0
        )
        kan_streamq_results = kan_streamq_agent.run(num_episodes=num_episodes, max_steps_per_episode=max_steps)
        print(f"KAN StreamQ Agent - Final Average Reward: {sum(kan_streamq_agent.total_rewards[-3:]) / 3:.2f}")

    return mlp_streamq_agent, kan_streamq_agent, mlp_streamq_results, kan_streamq_results


def compare_algorithms(basic_agents, streamq_agents):
    """Compare performance between basic and StreamQ algorithms."""
    mlp_basic, kan_basic = basic_agents
    mlp_streamq, kan_streamq = streamq_agents
    
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    
    print("\nFinal Performance Comparison (Last 3 Episodes Average):")
    print(f"{'Algorithm':<20} {'MLP Reward':<15} {'KAN Reward':<15}")
    print("-" * 50)
    
    mlp_basic_avg = sum(mlp_basic.total_rewards[-3:]) / 3 if len(mlp_basic.total_rewards) >= 3 else sum(mlp_basic.total_rewards) / len(mlp_basic.total_rewards)
    kan_basic_avg = sum(kan_basic.total_rewards[-3:]) / 3 if len(kan_basic.total_rewards) >= 3 else sum(kan_basic.total_rewards) / len(kan_basic.total_rewards)
    mlp_streamq_avg = sum(mlp_streamq.total_rewards[-3:]) / 3 if len(mlp_streamq.total_rewards) >= 3 else sum(mlp_streamq.total_rewards) / len(mlp_streamq.total_rewards)
    kan_streamq_avg = sum(kan_streamq.total_rewards[-3:]) / 3 if len(kan_streamq.total_rewards) >= 3 else sum(kan_streamq.total_rewards) / len(kan_streamq.total_rewards)
    
    print(f"{'Basic Q-Learning':<20} {mlp_basic_avg:<15.2f} {kan_basic_avg:<15.2f}")
    print(f"{'StreamQ(Lambda)':<20} {mlp_streamq_avg:<15.2f} {kan_streamq_avg:<15.2f}")
    
    print(f"\nMLP: StreamQ vs Basic = {mlp_streamq_avg - mlp_basic_avg:+.2f}")
    print(f"KAN: StreamQ vs Basic = {kan_streamq_avg - kan_basic_avg:+.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test StreamQ algorithms')
    parser.add_argument('--algorithm', '-a', choices=['basic', 'streamq', 'both'], 
                       default='both', help='Which algorithm to test')
    parser.add_argument('--env', '-e', default='CartPole-v1', 
                       help='Environment name')
    parser.add_argument('--episodes', '-n', type=int, default=100, 
                       help='Number of episodes to train')
    parser.add_argument('--max_steps', '-s', type=int, default=500, 
                       help='Maximum steps per episode')
    parser.add_argument('--network', choices=['mlp', 'kan', 'both'], 
                       default='both', help='Which network type to test')
    
    args = parser.parse_args()
    
    print(f"Running tests with:")
    print(f"  Environment: {args.env}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Network: {args.network}")
    
    basic_agents = None
    streamq_agents = None
    
    if args.algorithm in ['basic', 'both']:
        basic_agents = test_basic_streaming_agents(
            args.env, args.episodes, args.max_steps, args.network
        )
    
    if args.algorithm in ['streamq', 'both']:
        streamq_agents = test_streamq_lambda_agents(
            args.env, args.episodes, args.max_steps, args.network,
        )
    
    if args.algorithm == 'both' and basic_agents is not None and streamq_agents is not None:
        compare_algorithms(
            (basic_agents[0], basic_agents[1]), 
            (streamq_agents[0], streamq_agents[1])
        )

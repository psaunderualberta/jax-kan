"""
Sweep Configuration Templates for StreamQ Hyperparameter Optimization
"""

BASIC_STREAMING_SWEEP = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-2, 'distribution': 'log_uniform_values'},
        'discount_factor': {'values': [0.99]},
        'start_e': {'values': [1.0]},
        'end_e': {'min': 0.01, 'max': 0.2, 'distribution': 'uniform'},
        'decay_duration': {'values': [500, 1000, 2000, 5000]},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'values': [5, 7, 9, 11]},
        'kan_k': {'values': [2, 3, 4]},
        'kan_num_stds': {'values': [2, 3, 4, 5]},
        'seed': {'values': [0, 42, 123, 456, 789]}
    }
}

BASIC_QUICK_SWEEP = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.0001, 0.001, 0.01]},
        'discount_factor': {'values': [0.95, 0.99, 0.995]},
        'end_e': {'values': [0.01, 0.05, 0.1]},
        'decay_duration': {'values': [500, 1000, 2000]},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'values': [5, 7, 9]},
        'kan_k': {'values': [2, 3, 4]},
        'kan_num_stds': {'values': [2, 3, 4]},
        'seed': {'values': [0, 42, 123]}
    }
}

BASIC_COMPREHENSIVE_SWEEP = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-4, 'max': 1e-1, 'distribution': 'log_uniform_values'},
        'discount_factor': {"values": [0.99]},
        "start_e": {"values": [1.0]},
        "end_e": {"min": 0.01, "max": 0.2},
        "decay_duration": {"values": [500, 1000, 2000, 5000]},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'distribution': 'int_uniform','min': 3,'max': 12},
        'kan_k': {'distribution': 'int_uniform', 'min': 2,'max': 6},
        'kan_num_stds': {'distribution': 'int_uniform','min': 1,'max': 6},
        'seed': {'values': [0, 42, 123, 456, 789]}
    }
}

STREAMQ_QUICK_SWEEP = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.0001, 0.001, 0.01]},
        'discount_factor': {'values': [0.99]},
        'start_e': {'values': [1.0]},
        'end_e': {'values': [0.01, 0.05, 0.1]},
        "decay_duration": {'values': [500, 1000, 2000]},
        
        'lambda_': {'values': [0.5, 0.8, 0.9, 0.95]},
        'obs_std_init': {'values': [0.1, 0.5, 1.0]},
        'alpha_obs': {'values': [0.01, 0.05, 0.1]},
        'alpha_reward': {'values': [0.01, 0.05, 0.1]},
        'reward_std_init': {'values': [0.5, 1.0, 2.0]},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'values': [5, 7, 9]},
        'kan_k': {'values': [2, 3, 4]},
        'kan_num_stds': {'values': [2, 3, 4]},
        'seed': {'values': [0, 42, 123]}
    }
}

STREAMQ_COMPREHENSIVE_SWEEP = {
    'method': 'random',
    'metric': {'name': 'final_avg_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values','min': 1e-5,'max': 1e-2},
        'discount_factor': {'values': [0.99]},

        # Exploration parameters
        'start_e': {'values': [1.0]},
        'end_e': {'distribution': 'log_uniform_values','min': 0.01,'max': 0.2},
        'decay_duration': {'values': [500, 1000, 2000, 5000]},

        # StreamQ(Lambda) specific parameters
        'lambda_': {'distribution': 'uniform','min': 0.1,'max': 0.99},
        'obs_std_init': {'distribution': 'log_uniform_values','min': 0.1,'max': 2.0},
        'alpha_obs': {'distribution': 'uniform','min': 0.01,'max': 0.3},
        'alpha_reward': {'distribution': 'uniform','min': 0.01,'max': 0.3},
        'reward_std_init': {'distribution': 'log_uniform_values','min': 0.1,'max': 2.0},
        'hidden_dims': {"values": [
            (32,),
            (64,),
            (128,),
            (32, 32),
            (128, 128),
        ]},
        'kan_grid': {'distribution': 'int_uniform','min': 3,'max': 12},
        'kan_k': {'distribution': 'int_uniform', 'min': 2,'max': 6},
        'kan_num_stds': {'distribution': 'int_uniform','min': 1,'max': 6},
        'seed': {'values': [0, 42, 123, 456, 789]}
    }
}

ENVIRONMENTS = {
    'CartPole-v1': {
        'num_episodes': 200,
        'max_steps': 500,
        'success_threshold': 450
    },
    'MountainCar-v0': {
        'num_episodes': 300,
        'max_steps': 200,
        'success_threshold': -110
    },
    'Acrobot-v1': {
        'num_episodes': 250,
        'max_steps': 500,
        'success_threshold': -100
    }
}


SWEEP_CONFIGS = {
    'basic_quick': BASIC_QUICK_SWEEP,
    'basic_comprehensive': BASIC_COMPREHENSIVE_SWEEP,
    'streamq_quick': STREAMQ_QUICK_SWEEP,
    'streamq_comprehensive': STREAMQ_COMPREHENSIVE_SWEEP,
}

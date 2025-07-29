from kan import KAN
from mlp import MLP
from tabular import Table
from jax import random as jr, numpy as jnp, lax, value_and_grad, debug
import equinox as eqx
from tqdm import tqdm
import optax
from gymnax import make
from gymnax.environments import environment
from util import q_epsilon_greedy, q_huber_loss, linear_epsilon_schedule, mix_pytrees, sample_mean_var
import flashbax as fbx
from typing import Any
import chex
import wandb


class LoopState(eqx.Module):
    key: chex.PRNGKey
    network: MLP | KAN
    target_network: MLP | KAN
    opt_state: optax.GradientTransformation
    buffer_state: Any
    env_obs: chex.PRNGKey
    env_state: environment.Environment
    global_step: int = 0
    episode_num: int = 0


def main(conf=None):
    if conf is None:
        wandb.init()
        conf = wandb.config

    # Create Key
    key = jr.PRNGKey(conf.key)

    # Create environment
    env, env_params = make(conf.env_name)
    obs_dims = env.observation_space(env_params)
    obs_space = obs_dims.shape
    num_actions = env.num_actions
    training_episodes = conf.training_episodes
    gamma = 0.99

    # Create eps-decay settings
    start_e = conf.start_e
    end_e = conf.end_e
    decay_duration = conf.decay_duration
    tau = conf.tau

    # Initialize network
    key, _key = jr.split(key)
    dims = [obs_space[0], *conf.hidden_layers, num_actions]
    # network = KAN(dims, 7, 3, 3, _key)
    network = MLP(dims, _key)
    target_network = network
    # a = jnp.asarray([-2.4, -2.4, -0.21, -1.0])
    # network = Table(a, -a, 10, env.action_space(env_params).n, env_params.max_steps_in_episode)

    # optimizer
    batch_size = conf.batch_size
    optimizer = optax.adamw(conf.lr)
    opt_state = optimizer.init(network)

    # Create buffer
    buffer = fbx.make_item_buffer(max_length=conf.buffer_length, min_length=batch_size, sample_batch_size=batch_size)

    def warmup_buffer(loop_state: LoopState) -> LoopState:
        key, _key = jr.split(loop_state.key)
        eps = linear_epsilon_schedule(start_e, end_e, decay_duration * training_episodes, loop_state.episode_num)
        action, _, _ = q_epsilon_greedy(loop_state.network, loop_state.env_obs, eps, _key)

        key, _key = jr.split(key)
        next_obs, next_state, reward, done, _ = env.step(_key, loop_state.env_state, action, env_params)
        transition = {
            "obs": loop_state.env_obs,
            "reward": reward,
            "next_obs": next_obs,
            "action": action,
            "done": done,
        }

        buffer_state = buffer.add(loop_state.buffer_state, transition)

        return LoopState(
            key=key,
            network=loop_state.network,
            target_network=loop_state.target_network,
            buffer_state=buffer_state,
            env_obs=next_obs,
            env_state=next_state,
            opt_state=loop_state.opt_state,
            global_step=loop_state.global_step + 1,
            episode_num=loop_state.episode_num
        )


    def train_step(loop_state: LoopState) -> LoopState:
        key, _key = jr.split(loop_state.key)
        eps = linear_epsilon_schedule(start_e, end_e, decay_duration, loop_state.episode_num)
        action, _, _ = q_epsilon_greedy(loop_state.network, loop_state.env_obs, eps, _key)

        key, _key = jr.split(key)
        next_obs, next_state, reward, done, _ = env.step(_key, loop_state.env_state, action, env_params)

        key, _key = jr.split(key)
        transition = {
            "obs": loop_state.env_obs,
            "reward": reward,
            "next_obs": next_obs,
            "action": action,
            "done": done,
        }
        training_samples = buffer.sample(loop_state.buffer_state, _key).experience
        buffer_state = buffer.add(loop_state.buffer_state, transition)

        loss, grads = value_and_grad(q_huber_loss)(
            loop_state.network,
            training_samples['obs'],
            training_samples['action'],
            training_samples['reward'],
            training_samples['done'],
            training_samples['next_obs'],
            gamma,
            target_model=loop_state.target_network
        )

        # print((grads.q_values != 0).sum())
        # print(loss)
        # exit()

        # Update network
        updates, opt_state = optimizer.update(grads, loop_state.opt_state, loop_state.network)
        network = eqx.apply_updates(loop_state.network, updates)

        return LoopState(
            key=key,
            network=network,
            target_network=mix_pytrees(network, loop_state.target_network, tau=tau),
            buffer_state=buffer_state,
            env_obs=next_obs,
            env_state=next_state,
            opt_state=opt_state,
            global_step=loop_state.global_step + 1,
            episode_num=loop_state.episode_num + done.astype(loop_state.episode_num.dtype)
        )
    
    def eval_step(loop_state: LoopState) -> LoopState:
        current_episode_num = loop_state.episode_num
        bls = loop_state
        loop_state = lax.while_loop(
            lambda ls: ls.episode_num == current_episode_num,
            train_step,
            loop_state
        )

        return loop_state, loop_state.global_step

    key, _key = jr.split(key)
    obs, state = env.reset(key, env_params)
    dummy_transition = {
        "obs": obs,
        "action": 0,
        "reward": 0.0,
        "next_obs": obs,
        "done": False
    }

    buffer_state = buffer.init(dummy_transition)

    loop_state = LoopState(
        key=key,
        network=network,
        target_network=target_network,
        opt_state=opt_state,
        buffer_state=buffer_state,
        env_obs=obs,
        env_state=state
    )

    # Fill buffer
    loop_state = lax.while_loop(
        lambda ls: jnp.logical_not(buffer.can_sample(ls.buffer_state)),
        warmup_buffer,
        loop_state
    )   

    # Train for 'training_episodes' episodes
    # train_step(loop_state)
    loop_state, ls = lax.scan(
        lambda ls, _: eval_step(ls),
        loop_state,
        length=training_episodes,
    )

    episode_lengths = ls[1:] - ls[:-1]

    window_size = 20
    for i, length in enumerate(episode_lengths):
        if i >= window_size:
            window = episode_lengths[i-window_size:i]
        else:
            window = jnp.zeros(1)
    
        wandb.log({
            "length": length,
            "length mean":  window.mean(),
            "length var": window.var(),
        })

    wandb.finish()

    return loop_state


if __name__ == "__main__":

    # training_episodes = 10_000
    # gamma = 0.99

    # # Create eps-decay settings
    # start_e = 1.0
    # end_e = 0.01
    # decay_duration = 5_000
    # tau = 0.005

    # # Initialize network
    # key, _key = jr.split(key)
    # dims = [obs_space[0], 128, num_actions]
    # # network = KAN(dims, 7, 3, 3, _key)
    # network = MLP(dims, _key)
    # target_network = network
    # # a = jnp.asarray([-2.4, -2.4, -0.21, -1.0])
    # # network = Table(a, -a, 10, env.action_space(env_params).n, env_params.max_steps_in_episode)

    # # optimizer
    # batch_size = 128
    # optimizer = optax.adamw(1e-4)
    # opt_state = optimizer.init(network)

    # # Create buffer
    # buffer = fbx.make_item_buffer(max_length=10_000, min_length=batch_size, sample_batch_size=batch_size)

    mlp_sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "length mean"},
        "parameters": {
            "env_name": {"values": ["CartPole-v1"]},
            "key": {"min": 0, "max": 10**6, "distribution": "int_uniform"},
            "batch_size": {"values": [16, 32, 64, 128, 512]},
            "buffer_length": {"values": [512, 1024, 4096, 10_000]},
            "training_episodes": {"values": [5_000, 10_000, 50_000, 100_000]},
            "lr": {"max": 1e-2, "min": 1e-5, "distribution": "log_uniform_values"},
            "hidden_layers": {"values": [
                (32,),
                (64,),
                (128,),
                (32, 32),
                (128, 128),
            ]},
            "start_e": {"values": [1.0]},
            "end_e": {"min": 0.01, "max": 0.2},
            "decay_duration": {"min": 0.5, "max": 0.95},
            "tau": {"min": 0.0005, "max": 0.1, "distribution": "log_uniform_values"}
        },
    }

    sweep_id = wandb.sweep(sweep=mlp_sweep_configuration, entity="kan_rl", project="Buffer-test")
    wandb.agent(sweep_id=sweep_id, function=main, count=500)
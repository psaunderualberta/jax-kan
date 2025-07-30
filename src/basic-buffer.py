from kan import KAN
from mlp import MLP
from tabular import Table
from jax import random as jr, numpy as jnp, lax, value_and_grad, debug, vmap
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
import argparse
from pprint import pprint
import pandas as pd


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


def setup_sweep(conf):
    mlp_sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "reward-mean"},
        "parameters": {
            "env_name": {"values": [conf.env_name]},
            "key": {"min": 0, "max": 10**6, "distribution": "int_uniform"},
            "batch_size": {"values": conf.batch_size},
            "buffer_length": {"values": conf.buffer_length},
            "training_steps": {"values": conf.training_steps},
            "num_evals": {"values": conf.num_evals},
            "lr": {"max": 1e-2, "min": 1e-5, "distribution": "log_uniform_values"},
            "hidden_layers": {"values": conf.hidden_layers},
            "start_e": {"values": [conf.start_e]},
            "end_e": {"min": conf.end_e_min, "max": conf.end_e_max},
            "decay_duration": {"min": conf.decay_duration_min, "max": conf.decay_duration_max},
            "tau": {"min": conf.tau_min, "max": conf.tau_max, "distribution": "log_uniform_values"}
        },
    }

    sweep_id = wandb.sweep(sweep=mlp_sweep_configuration, entity="kan_rl", project="Buffer-test")
    wandb.agent(sweep_id=sweep_id, function=main, count=conf.num_runs)


def main(conf=None):
    if conf is None:  # Only when calling 'main' via wandb sweep
        wandb.init()
        conf = wandb.config
    elif conf.wandb:
        wandb.init(entity="kan_rl", project="Buffer-test", config=conf)
    
    # Create Key
    key = jr.PRNGKey(conf.key)

    # Create environment
    env, env_params = make(conf.env_name)
    obs_dims = env.observation_space(env_params)
    obs_space = obs_dims.shape
    num_actions = env.num_actions
    training_steps = conf.training_steps
    num_evals = conf.num_evals
    eval_step_size = training_steps // num_evals
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

    def evaluate(network, key):
        key, _key = jr.split(key)
        obs, state = env.reset(_key, env_params)
        cum_reward = 0.0
        carry = (obs, state, key, cum_reward, 0.0, False)

        def body(carry):
            obs, state, key, cum_reward, length, _ = carry
            key, action_key, step_key = jr.split(key, 3)
            action, _, _ = q_epsilon_greedy(network, obs, 0.0, action_key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)

            return (
                next_obs,
                next_state,
                key,
                cum_reward + gamma**length * reward,
                length + 1,
                done
            )
        
        (obs, state, key, cum_reward, length, done) = lax.while_loop(
            lambda c: jnp.logical_not(c[5]),
            body,
            carry
        )

        return cum_reward, length


    def warmup_buffer(loop_state: LoopState) -> LoopState:
        key, _key = jr.split(loop_state.key)
        eps = linear_epsilon_schedule(start_e, end_e, decay_duration * training_steps, loop_state.global_step)
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
        ), None
    
    def eval_step(loop_state: LoopState) -> LoopState:
        loop_state, _ = lax.scan(
            lambda ls, _: train_step(ls),
            loop_state,
            length=eval_step_size
        )

        eval_keys = jr.split(loop_state.key, 50)

        vmapped_eval = vmap(evaluate, in_axes=(None, 0))
        evaluations = vmapped_eval(loop_state.network, eval_keys)

        debug.print("Global Step: {}, Ep: {}, Avg. Reward: {:.2f}, Avg. Length: {:.2f}",
                    loop_state.global_step,
                    loop_state.episode_num,
                    evaluations[0].mean(),
                    evaluations[1].mean()
        )

        return loop_state, evaluations

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
    loop_state, evaluations = lax.scan(
        lambda ls, _: eval_step(ls),
        loop_state,
        length=num_evals,
    )

    rewards, lengths = evaluations

    df = pd.DataFrame(columns=["Reward", "Length"])

    for i in range(rewards.shape[0]):
        ep_rewards = rewards[i, :]
        ep_lengths = lengths[i, :]

        df.loc[len(df)] = {
            "Reward": ep_rewards.tolist(),
            "Length": ep_lengths.tolist(),
        }
    
        if conf.wandb:
            wandb.log({
                "reward-mean": ep_rewards.mean(),
                "reward-var": ep_rewards.var(),
                "length-mean":  ep_lengths.mean(),
                "length-var": ep_lengths.var(),
            }, step=eval_step_size * (i + 1))

    if conf.wandb:
        wandb.log({"df": wandb.Table(dataframe=df)})
        wandb.finish()

    return loop_state


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Subcommand Help")

    ## Single run parser
    run_parser = subparsers.add_parser("run", help="Run Help")
    run_parser.add_argument("--env_name", default="CartPole-v1", type=str, help="The RL Environment")
    run_parser.add_argument("--key", default=0, type=int, help="The JAX PRNG key")
    run_parser.add_argument("--batch_size", default=512, type=int, help="The replay batch size")
    run_parser.add_argument("--buffer-length", default=10_000, type=int, help="The Buffer Replay Length")
    run_parser.add_argument("--training-steps", default=10_000, type=int, help="The # of steps to train")
    run_parser.add_argument("--num-evals", default=10_000, type=int, help="The # of evaluation steps, evenly spaced")
    run_parser.add_argument("--lr", default=1e-3, type=float, help="The Learning Rate")
    run_parser.add_argument("--hidden_layers", default=(32,), type=tuple[int])
    run_parser.add_argument("--start_e", default=1.0, type=float, )
    run_parser.add_argument("--end_e", default=0.01, type=float, )
    run_parser.add_argument("--decay_duration", default=0.5, type=float)
    run_parser.add_argument("--tau", default=5e-4, help="The mixing parameter for updating the target network. ")
    run_parser.add_argument("--wandb", action="store_true")
    run_parser.set_defaults(func=main) # Directly call 'main' 

    # sweep parser
    run_parser = subparsers.add_parser("sweep", help="Run Help")
    run_parser.add_argument("--env_name", default="CartPole-v1", type=str, help="The RL Environment", nargs="+")
    run_parser.add_argument("--num_runs", default=50, type=int, help="The number of runs within the sweep", nargs="+")
    run_parser.add_argument("--batch-size", default=(16, 32, 64, 128, 512), type=tuple[int], help="The replay batch size", nargs="+")
    run_parser.add_argument("--buffer-length", default=(512, 1024, 4096, 10_000), type=tuple[int], help="The Buffer Replay Length", nargs="+")
    run_parser.add_argument("--training-steps", default=(600_000,), type=tuple[int], help="The # of steps to train", nargs="+")
    run_parser.add_argument("--num-evals", default=(250,), type=int, help="The # of evaluation steps, evenly spaced")
    run_parser.add_argument("--lr-min", default=1e-5, type=float, help="The Learning Rate Minimum")
    run_parser.add_argument("--lr-max", default=1e-2, type=float, help="The Learning Rate Maximum")
    run_parser.add_argument("--hidden_layers", default=((32,),(64,),(128,),(32, 32),(128, 128)), type=tuple[int], nargs="+")
    run_parser.add_argument("--start_e", default=1.0, type=float)
    run_parser.add_argument("--end_e-min", default=0.01, type=float)
    run_parser.add_argument("--end_e-max", default=0.2, type=float)
    run_parser.add_argument("--decay_duration-min", default=0.5, type=float)
    run_parser.add_argument("--decay_duration-max", default=0.95, type=float)
    run_parser.add_argument("--tau-min", default=5e-4, help="The minimum mixing parameter for updating the target network. ")
    run_parser.add_argument("--tau-max", default=0.1, help="The maximum mixing parameter for updating the target network. ")
    run_parser.set_defaults(func=setup_sweep) # Directly call 'main' 

    args = parser.parse_args()
    args.func(args)

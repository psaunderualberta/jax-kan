from util import DATALOADERS
from kan import KAN
from mlp import MLP
from jax import random as jr, numpy as jnp, lax
from util import non_vmap_cce_loss, subset_classification_accuracy, value_and_grad
import equinox as eqx
from tqdm import tqdm
import optax
from gymnax import make
from gymnax.environments import environment
from util import q_epsilon_greedy, q_huber_loss
from experiments.buffered import Transition, Buffer, BufferState
import chex


class LoopState(eqx.Module):
    key: chex.PRNGKey
    network: MLP | KAN
    opt_state: optax.GradientTransformation
    buffer: BufferState
    env_obs: chex.PRNGKey
    env_state: environment.Environment


def main():
    # Create Key
    key = jr.PRNGKey(0)

    # Create environment
    env, env_params = make("CartPole-v1")
    obs_space = env.observation_space(env_params).shape
    num_actions = env.num_actions
    gamma = 0.99


    # Create buffer
    buffer = BufferState.create(10000)


    # Initialize network
    key, _key = jr.split(key)
    dims = [obs_space[0], 32, num_actions]
    network = KAN(dims, 7, 3, 3, _key)

    # optimizer
    batch_size = 512
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(network)

    def warmup_buffer(loop_state: LoopState) -> LoopState:
        key, _key = jr.split(loop_state.key)
        action = q_epsilon_greedy(loop_state.network, loop_state.env_obs, eps, _key)

        key, _key = jr.split(key)
        next_obs, next_state, reward, done, _ = env.step(_key, loop_state.env_state, action, env_params)

        transition = Transition(obs=loop_state.obs, action=action, next_obs=next_obs, done=done)
        buffer = Buffer.push(buffer, transition)

        return LoopState(
            key=key,
            network=loop_state.network,
            buffer=buffer,
            env_obs=next_obs,
            next_state=next_state
        )


    def train_step(loop_state: LoopState) -> LoopState:
        key, _key = jr.split(loop_state.key)
        action = q_epsilon_greedy(loop_state.network, loop_state.env_obs, eps, _key)

        key, _key = jr.split(key)
        next_obs, next_state, reward, done, _ = env.step(_key, loop_state.env_state, action, env_params)

        transition = Transition(obs=loop_state.obs, action=action, next_obs=next_obs, done=done)
        buffer = Buffer.push(buffer, transition)

        training_samples = buffer.sample(batch_size)

        loss, grads = value_and_grad(q_huber_loss)(
            loop_state.network,
            loop_state.env_obs,
            action,
            reward,
            done,
            next_obs,
            gamma,
        )

        # Update network
        updates, opt_state = optimizer.update(grads, loop_state.opt_state, network)
        network = eqx.apply_updates(network, updates)

        return LoopState(
            key=key,
            network=loop_state.network,
            buffer=buffer,
            env_obs=next_obs,
            next_state=next_state,
            opt_state=opt_state
        )

    key, _key = jr.split(key)
    obs, state = env.reset(key, env_params)
    loop_state = LoopState(
        key=key,
        network=network,
        opt_state=opt_state,
        buffer=buffer,
        env_obs=obs,
        env_state=state
    )

    # Fill buffer
    loop_state = lax.while_loop(
        lambda ls: jnp.logical_not(ls.buffer.done),
        warmup_buffer,
        loop_state
    )


if __name__ == "__main__":
    main()

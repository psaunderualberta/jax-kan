from tabular import Table
import jax.numpy as jnp
from gymnax import make


def main():
    env, env_params = make("CartPole-v1")

    obs_dims = env.observation_space(env_params)

    table = Table(obs_dims.low, obs_dims.high, 50, env.action_space(env_params).n)
    print(table.obs_grid)
    table(jnp.array([0., 0., 0., 0.]))
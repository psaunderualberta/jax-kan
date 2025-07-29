import equinox as eqx
import jax.numpy as jnp
from gymnax import make
import chex


class Table(eqx.Module):
    obs_grid: chex.Array
    q_values: chex.Array


    def __init__(self, obs_low, obs_high, obs_resolution: int, num_actions: int, init_value: float = 0.0):
        self.obs_grid = jnp.linspace(obs_low, obs_high, obs_resolution)

        shape = (obs_resolution,) * len(obs_low)
        self.q_values = jnp.ones((*shape, num_actions)) * init_value

    def __call__(self, x):
        idxs = (x < self.obs_grid).argmax(axis=0)
        return self.q_values[tuple(idxs)]

    def num_actions(self):
        return self.q_values.shape[-1]
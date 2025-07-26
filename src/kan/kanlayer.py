import equinox as eqx
from util import SILU
import jax.numpy as jnp
import jax.random as jr
import chex


class KANLayer(eqx.Module):
    def __init__(
        self, in_dim: int, out_dim: int,
        grid: int, k: int, num_stds: int, 
        key: chex.PRNGKey
    ):
        limit = 1 / jnp.sqrt(in_dim)
        self.w_b = jr.uniform(key, (in_dim, out_dim), minval=-lim, maxval=lim)
        self.w_s = jr.uniform(key, (in_dim, out_dim), minval=-lim, maxval=lim)

        grid_width = grid / (2 * num_stds)
        grid_endpoint = k * grid_width + grid

        self.num_splines = grid + k
        self.grid_points = jnp.linspace(-grid_endpoint, grid_endpoint, grid + 2 * k)

        self.coeffs = jr.uniform(key, (in_dim, out_dim, self.num_splines), minval=-lim, maxval=lim)
        

    def __call__(self):
        pass
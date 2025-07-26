import equinox as eqx
from util import SILU
import jax.numpy as jnp
import jax.random as jr
from splines.bspline import bspline
import chex


class KANLayer(eqx.Module):
    w_b: chex.Array = None
    w_s: chex.Array = None

    num_splines: int = 0
    num_control_points: int = 0

    grid_points: chex.Array = None
    control_points: chex.Array = None

    def __init__(
        self, in_dim: int, out_dim: int,
        grid: int, k: int, num_stds: int, 
        key: chex.PRNGKey
    ):
        limit = 1 / jnp.sqrt(in_dim)
        self.w_b = jr.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
        self.w_s = jr.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)

        grid_width = grid / num_stds
        grid_endpoint = k * grid_width + grid

        num_knots = grid + 1
        num_augmented_knots = num_knots + 2 * k
        self.num_splines = grid + k
        self.grid_points = jnp.linspace(-k * grid_width, grid + k * grid_width, num_augmented_knots)

        self.num_control_points = grid + k - 1
        self.control_points = jr.uniform(key, (in_dim, out_dim, self.num_control_points), minval=-limit, maxval=limit)
        

    def __call__(self, x):
        


if __name__ == "__main__":
    pass
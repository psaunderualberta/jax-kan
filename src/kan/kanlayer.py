import equinox as eqx
from util import SILU
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from jax import vmap
from splines import bspline_multi_control
import chex


class KANLayer(eqx.Module):
    w_b: chex.Array = None
    w_s: chex.Array = None

    grid_points: chex.Array = None
    control_points: chex.Array = None

    silu: eqx.Module = None
    bound: float = 0

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid: int,
        k: int,
        num_stds: int,
        key: chex.PRNGKey,
    ):

        # initialization - ones for bias, xavier for matrix
        # see https://arxiv.org/pdf/2404.19756, page 6
        self.w_b = jnp.ones((in_dim, out_dim))
        limit = jnp.sqrt(6 / (in_dim + out_dim))
        self.w_s = jr.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)

        # Grid
        grid_width = grid / num_stds
        num_knots = grid + 1
        num_augmented_knots = num_knots + 2 * k
        self.grid_points = jnp.linspace(
            -k * grid_width, grid + k * grid_width, num_augmented_knots
        )

        # Coefficients, i.e. control points in spline terminology
        num_control_points = grid + k - 1
        self.control_points = jr.uniform(
            key, (in_dim, out_dim, num_control_points), minval=-limit, maxval=limit
        )

        # activation
        self.silu = SILU()
        self.bound = num_stds * 1.0

    def __call__(self, x):
        x = x.T

        # clip x to within appropriate range
        bound = lax.stop_gradient(self.bound)
        x = jnp.clip(x, min=-bound, max=bound)

        # (in * out) x (in * num_datapoints) -> (in * out * num_datapoints)
        biases = jnp.einsum("ij,ik->ijk", self.w_b, self.silu(x))  # TODO: Correct?

        # (in * out * coeff) x (in * num_datapoints) -> (out * num_datapoints)
        stable_grid_points = lax.stop_gradient(self.grid_points)
        vmapped_bspline = vmap(bspline_multi_control, in_axes=(0, None, 0, None))
        mapped = vmapped_bspline(x, stable_grid_points, self.control_points, 3)

        non_summed_activations = biases + self.w_s[:, :, None] * mapped
        summed_activations = jnp.sum(non_summed_activations, axis=0)

        return summed_activations.T


if __name__ == "__main__":
    pass

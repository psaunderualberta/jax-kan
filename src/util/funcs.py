import equinox as eqx
import jax.numpy as jnp


class SILU(eqx.Module):
    def __call__(self, x):
        return x / (1 + jnp.exp(-x))

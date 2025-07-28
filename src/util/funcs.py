import equinox as eqx
import jax.numpy as jnp


class SILU(eqx.Module):
    def __call__(self, x):
        return x / (1 + jnp.exp(-x))


def linear_epsilon_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return  jnp.maximum(slope * t + start_e, end_e)
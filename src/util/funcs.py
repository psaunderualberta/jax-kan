import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt


class SILU(eqx.Module):
    def __call__(self, x):
        return x / (1 + jnp.exp(-x))


def linear_epsilon_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return  jnp.maximum(slope * t + start_e, end_e)


def mix_pytrees(a, b, tau=0.005):
    def mix(a_, b_):
        if a_ is None:
            return a_
        
        return a_ * tau + b_ * (1 - tau)
    
    return jt.map(mix, a, b, is_leaf=lambda x: x is None)
import equinox as eqx
from kan import KANLayer
import jax.random as jr
import jax.numpy as jnp
import chex


class KAN(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self, dims: list[int],
        grid: int, k: int, num_stds: int, 
        key: chex.PRNGKey
    ):
        in_dim = dims[0]
        layers = []
        for out_dim in dims[1:-1]:
            key, _key = jr.split(key)
            layers.append(KANLayer(
                in_dim=in_dim, out_dim=out_dim, grid=grid, k=k, num_stds=num_stds, key=_key
            ))

            # move to next layer
            in_dim=out_dim
        
        layers.append(KANLayer(
            in_dim=in_dim, out_dim=dims[-1], grid=grid, k=k, num_stds=num_stds, key=_key
        ))

        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:1]:
            x = layer(x)
            x = (x - x.mean(axis=1, keepdims=True)) / jnp.sqrt(x.var(axis=1, keepdims=True) + 1e-5)
        
        return self.layers[-1](x)
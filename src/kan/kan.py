import equinox as eqx
from kan import KANLayer
import jax.random as jr
import jax.numpy as jnp
import chex


class KAN(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self, dims: list[int], grid: int, k: int, num_stds: int, key: chex.PRNGKey
    ):
        in_dim = dims[0]
        self.layers = []
        for out_dim in dims[1:]:
            key, _key = jr.split(key)
            self.layers.append(
                KANLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid=grid,
                    k=k,
                    num_stds=num_stds,
                    key=_key,
                )
            )

            # move to next layer
            in_dim = out_dim

    def __call__(self, x):
        for layer in self.layers[:1]:
            # Linear layer
            x = layer(x)

            # Layer normalization to ensure values stay within appropriate
            # bounds (i.e. spline range)
            x = (x - x.mean(keepdims=True)) / jnp.sqrt(
                x.var(keepdims=True) + 1e-5
            )

        # Final layer, no ultimate normalization
        return self.layers[-1](x)

    def num_actions(self):
        return self.layers[-1].w_b.shape[0]
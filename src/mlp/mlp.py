import equinox as eqx
import jax.random as jr
import jax.nn
from util import SILU
import chex


class MLP(eqx.Module):
    layers: list[eqx.Module]

    def __init__(self, dims: chex.Array, key: chex.PRNGKey, activation: eqx.Module = SILU):
        in_dim = dims[0]
        self.layers = []
        for out_dim in dims[1:-1]:
            key, _key = jr.split(key)
            self.layers.append(
                eqx.nn.Linear(
                    in_dim,
                    out_dim,
                    key=_key
                )
            )

            self.layers.append(activation())

            # move to next layer
            in_dim = out_dim
        
        self.layers.append(
            eqx.nn.Linear(in_dim, dims[-1], key=key)
        )

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        
        return self.layers[-1](x)
    
    def num_actions(self):
        return self.layers[-1].in_features

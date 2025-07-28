import equinox as eqx
import jax.random as jr
import chex


class MLP(eqx.Module):
    def __init__(self, dims: chex.Array, key: chex.PRNGKey):
        in_dim = dims[0]
        self.layers = []
        for out_dim in dims[1:]:
            key, _key = jr.split(key)
            self.layers.append(
                eqx.nn.Linear(
                    in_dim,
                    out_dim,
                    key=_key
                )
            )

            # move to next layer
            in_dim = out_dim

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x

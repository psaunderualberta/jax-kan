import equinox as eqx
from kan import KANLayer


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
            layers.append(KANLayer(
                in_dim=in_dim, out_dim=out_dim, grid=grid, k=k, num_stds=num_stds, key=_key
            ))

            # add layer normalization to ensure the method is within the grid
            layers.append(eqx.nn.LayerNorm(out_dim, use_weight=False, use_bias=False))

            # move to next layer
            in_dim=out_dim
        
        layers.append(KANLayer(
            in_dim=in_dim, out_dim=dims[-1], grid=grid, k=k, num_stds=num_stds, key=_key
        ))

        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
from util import DATALOADERS
from kan import KANLayer
from jax import random as jr, numpy as jnp


def main():
    layer = KANLayer(in_dim=2, out_dim=3, grid=5, k=3, num_stds=5, key=jr.PRNGKey(0))
    x = jnp.arange(10).reshape(2, -1) / 5
    print(x)
    layer(x)


if __name__ == "__main__":
    main()

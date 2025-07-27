from util import DATALOADERS
from kan import KAN
from jax import random as jr, numpy as jnp
from util import non_vmap_cce_loss
import optax


def main():
    # Load data
    x, y = DATALOADERS["mnist"]()

    dims = [x.shape[1], 32, y.shape[1]]

    network = KAN(dims, 7, 3, 3, jr.PRNGKey(0))

    dataset_idxs = jnp.arange(x.shape[0])
    shuffle_key = jr.PRNGKey(0)

    batch_size = 512
    batch_x, batch_y = x[:32, :], y[:32, :]
    optimizer = optax.sgd(learning_rate=0.003)
    opt_state = optimizer.init(network)

    while True:
        shuffle_key, _key = jr.split(shuffle_key)
        idxs = jr.permutation(_key, x.shape[0])[:batch_size]

        batch_x, batch_y = x[idxs, :], y[idxs, :]
        print(batch_x.shape)

        loss, grads = non_vmap_cce_loss(network, batch_x, batch_y)
        print(round(loss, 3))
        updates, opt_state = optimizer.update(grads, opt_state, network)
        network = optax.apply_updates(network, updates)


if __name__ == "__main__":
    main()

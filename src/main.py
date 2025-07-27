from util import DATALOADERS
from kan import KAN
from jax import random as jr, numpy as jnp
from util import non_vmap_cce_loss, subset_classification_accuracy
import equinox as eqx
from tqdm import tqdm
import optax


def main():
    # Load data
    x, y = DATALOADERS["mnist"]()
    key = jr.PRNGKey(0)
    key, _key = jr.split(key)

    # Initialize network
    dims = [x.shape[1], 32, y.shape[1]]
    network = KAN(dims, 7, 3, 3, _key)

    # optimizer
    batch_size = 512
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(network)

    # Create visual indicator of progress
    iterator = tqdm(range(1000), total=1000)
    for iteration in iterator:
        # Get batch, note that we don't
        # iterate over the entire dataset then shuffle.
        # Easier to implement this way :P
        key, _key = jr.split(key)
        idxs = jr.permutation(_key, x.shape[0])[:batch_size]
        batch_x, batch_y = x[idxs, :], y[idxs, :]

        # Compute loss, accuracy, gradients
        key, _key = jr.split(key)
        loss, grads = non_vmap_cce_loss(network, batch_x, batch_y)
        accuracy = subset_classification_accuracy(network, x, y, 0.01, _key)

        # Update iterator description
        iterator.set_description(f"{iteration} | {loss:.4f} | {accuracy:.2f}")

        # Update network
        updates, opt_state = optimizer.update(grads, opt_state, network)
        network = eqx.apply_updates(network, updates)  # None-safe


if __name__ == "__main__":
    main()

import equinox as eqx
from optax import softmax_cross_entropy
from jax import vmap, numpy as jnp, random as jr
import chex


@eqx.filter_jit
@eqx.filter_value_and_grad
def non_vmap_cce_loss(model: eqx.Module, x: chex.Array, y: chex.Array):
    pred_y = model(x).squeeze()

    return softmax_cross_entropy(pred_y, y).mean()

@eqx.filter_jit
def classification_accuracy(model, x, y):
    pred_y = model(x)
    pred_y_int = jnp.argmax(pred_y, axis=1)
    y_int = jnp.argmax(y, axis=1)
    return jnp.mean(pred_y_int == y_int) * 100


def subset_classification_accuracy(model, x, y, percent, key):
    num_samples = x.shape[0]
    num_samples = int(num_samples * percent)
    idxs = jr.permutation(key, jnp.arange(x.shape[0]))[:num_samples]
    return classification_accuracy(model, x[idxs], y[idxs])
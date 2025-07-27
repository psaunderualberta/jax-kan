import equinox as eqx
from optax import softmax_cross_entropy


@eqx.filter_jit
@eqx.filter_value_and_grad
def non_vmap_cce_loss(model: eqx.Module, x: chex.Array, y: chex.Array):
    pred_y = model(x).squeeze()

    return softmax_cross_entropy(pred_y, y).mean()

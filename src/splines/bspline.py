from jax import jit, vmap, numpy as jnp
import chex
from functools import partial


@partial(jit, static_argnames=('order',))
def bspline(x: chex.Array, knots: chex.Array, controls: chex.Array, order: int):
    """Evaluates a B-spline using de Boor's algorithm
    See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for more info

    Arguments
    ---------
    x: Elements at which to evaluate spline(x)
    knots: Array of knot positions, *not yet padded*
    controls: Array of control points. 
    order: Degree of B-spline.
    """
    idxs = jnp.searchsorted(knots, x, side='right') - 1
    d = [controls[j + idxs - order] for j in range(0, order + 1)]

    for r in range(1, order + 1):
        for j in range(order, r - 1, -1):
            alpha = (x - knots[j + idxs - order]) / (knots[j + 1 + idxs - r] - knots[j + idxs - order])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    
    return d[order]


if __name__ == "__main__":
    knots = jnp.asarray([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    p = 3
    print(knots)

    controls = jnp.asarray([1, 1, 1, 1, 1, 1, 1])
    # x = jnp.asarray([-2, -1, 0, 1, 2])
    x = jnp.asarray([-2, -1, 0, 1, 1.99])
    print(bspline(x, knots, controls, p))
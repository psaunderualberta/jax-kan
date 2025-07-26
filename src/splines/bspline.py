from jax import jit, vmap, numpy as jnp
import chex
from functools import partial


@partial(vmap, in_axes=(0, None))
def parallel_vmap(a, v):
    return jnp.searchsorted(a, v, side='right')


@partial(jit, static_argnames=('order',))
def bspline(x: chex.Array, knots: chex.Array, controls: chex.Array, order: int):
    """Evaluates a B-spline using de Boor's algorithm
    See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for more info

    Arguments
    ---------
    x: Elements at which to evaluate spline(x)
    knots: Array of knot positions, needs to be padded as described above.
    controls: Array of control points. 
    order: Degree of B-spline.
    """
    
    idxs = parallel_vmap(x, knots)
    d = [controls[j + idxs - order] for j in range(0, order + 1)] 

    for r in range(1, order + 1):
        for j in range(order, r - 1, -1):
            alpha = (x - knots[j + idxs - order]) / (knots[j + 1 + idxs - r] - knots[j + idxs - order]) 
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    
    return d


if __name__ == "__main__":
    knots = jnp.asarray([1, 2, 3])
    p = 3
    print(knots)
    knots = jnp.pad(knots, p, mode='edge')
    print(knots)
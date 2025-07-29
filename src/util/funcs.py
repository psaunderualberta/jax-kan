import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jax import lax as jax_lax, tree_util as jtu, jit
from jax import random as jr
import chex

import equinox as eqx
from typing import Union

__eps = 1e-5


def get_float_dtype():
    """Returns the default float dtype."""
    return jnp.float32


class SILU(eqx.Module):
    def __call__(self, x):
        return x / (1 + jnp.exp(-x))


def linear_epsilon_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return  jnp.maximum(slope * t + start_e, end_e)


def mix_pytrees(a, b, tau=0.005):
    def mix(a_, b_):
        if a_ is None:
            return a_
        
        return a_ * tau + b_ * (1 - tau)
    
    return jt.map(mix, a, b, is_leaf=lambda x: x is None)


def sample_mean_var(statistic, mu, p, count):
    """
    See the 'Stream-Q' paper. Online computation of running mean and variance. 
    """
    count += 1
    mu_bar = mu + (statistic - mu) / count
    p = p + (statistic - mu) * (statistic - mu_bar)
    var = 1 if count < 2 else p / (count - 1)
    return mu_bar, var, count


# the following utils were taken from this source code:
# source: https://github.com/psaunderualberta/stream-jax/blob/main/src/util.py
@eqx.filter_jit
def is_none(x):
    return x is None
    

@eqx.filter_jit
def ObGD(
    eligibility_trace: chex.Array,
    model: chex.Array,
    delta: Union[float, chex.Array],
    alpha: Union[float, chex.Array],
    kappa: Union[float, chex.Array]
):
    delta_bar = jnp.maximum(jnp.abs(delta), 1.0)

    eligibility_trace_norm = 0
    for leaf in jtu.tree_leaves(eligibility_trace):
        eligibility_trace_norm += jnp.sum(jnp.abs(leaf))

    M = alpha * kappa * delta_bar * eligibility_trace_norm
    alpha_ = jnp.minimum(alpha / M, alpha)

    # update in direction of gradient
    def _apply_update(m, e):
        if e is None:
            return m
        return m - alpha_ * delta * e

    return jtu.tree_map(_apply_update, model, eligibility_trace)


class SampleMeanStats(eqx.Module):
    mu: chex.Array
    p: chex.Array
    var: chex.Array
    count: int

    def __init__(self, mu, p, var, count):
        self.mu = mu
        self.p = p
        self.var = var
        self.count = count

    @classmethod
    def new_params(cls, shape):
        mu = jnp.zeros(shape, dtype=get_float_dtype())
        p = jnp.ones(shape, dtype=get_float_dtype())
        var = jnp.ones(shape, dtype=get_float_dtype())
        count = 1

        return SampleMeanStats(
            mu=mu,
            p=p,
            var=var,
            count=count,
        )

class SampleMeanUpdate(eqx.Module):
    @classmethod
    def update(cls, sample: Union[float, chex.Array], stats: SampleMeanStats):
        new_count = stats.count + 1
        mu_bar = stats.mu + (sample - stats.mu) / new_count
        new_p = stats.p + (sample - stats.mu) * (sample - mu_bar)
        var = jax_lax.select(new_count >= 2, new_p / (new_count - 1), jnp.ones_like(new_p))
        return SampleMeanStats(
            mu=mu_bar,
            p=new_p,
            var=var,
            count=new_count
        )

def normalize_observation(
    observation: chex.Array,
    observation_stats: SampleMeanStats
):
    new_stats = SampleMeanUpdate.update(observation, observation_stats)
    return (observation - new_stats.mu) / jnp.sqrt(new_stats.var + __eps), new_stats


@jit
def scale_reward(
    reward: Union[float, chex.Array],
    reward_stats: SampleMeanStats,
    reward_trace: Union[float, chex.Array],
    done: bool,
    gamma: Union[float, chex.Array],
):
    reward_trace = gamma * (1 - done) * reward_trace + reward
    new_stats = SampleMeanUpdate.update(reward_trace, reward_stats)
    reward_scaled = reward / jnp.sqrt(new_stats.var + __eps)
    return reward_scaled, reward_trace, new_stats


# @eqx.filter_jit
def sparse_init_linear(in_size, out_size, sparsity_level, key):
    layer_size = (out_size, in_size)  # equinox expects (out_size, in_size) for weight shape
    init_bound = 1 / jnp.sqrt(in_size)

    # Generate a random mask for sparsity
    zeros_per_col = jnp.ceil(sparsity_level * in_size).astype(int)

    # Initialize weights with lecun initialization + sparsity
    key, key_ = jr.split(key)
    weights = jr.uniform(key_, layer_size, get_float_dtype(), minval=-init_bound, maxval=init_bound)

    # init same as source code
    key, key_ = jr.split(key)
    zeros = jr.bernoulli(key_, 1 - sparsity_level, weights.shape)
    weights = weights * zeros

    # Initialize bias
    bias = jnp.zeros((out_size,), dtype=get_float_dtype())
    return weights, bias


def update_eligibility_trace(
    z_w,
    gamma, 
    lambda_,
    new_term
):
    def update_trace(z_w_, new_term_):
        if new_term_ is None:
            return z_w_
        return gamma * lambda_ * z_w_ + new_term_
    return jtu.tree_map(update_trace, z_w, new_term)


def pytree_if_else(
    pred,
    pt1,
    pt2,
    is_leaf=is_none
):
    def body(l, r):
        if l is None:
            return l
        return jax_lax.select(pred, l, r)
    
    return jtu.tree_map(body, pt1, pt2, is_leaf=is_leaf)


@eqx.filter_jit
def init_eligibility_trace(
    model: eqx.Module
):
    def fun(model_arr):
        if model_arr is None:
            return model_arr
        return jnp.zeros_like(model_arr)

    return jtu.tree_map(fun, model, is_leaf=is_none)

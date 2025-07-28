import equinox as eqx
from jax import numpy as jnp
from jax import lax
from jax import random as jr
import chex
from typing import Any


class BufferState(eqx.Module):
    buffer: chex.Array
    capacity: int
    full: bool = False
    ptr: int = 0

    @classmethod
    def create(cls, capacity) -> "BufferState":
        return cls(
            buffer=[None for _ in range(capacity)],
            capacity=capacity,
        )


class Buffer(eqx.Module):
        
    @classmethod
    def push(cls, state: BufferState, el: Any) -> BufferState:
        buffer = state.buffer
        ptr = state.ptr

        buffer[ptr] = el
        ptr = (ptr + 1) % state.capacity
        done = state.done or ptr == 0

        return BufferState(
            buffer=buffer,
            capacity=state.capacity,
            ptr=ptr,
            done=done
        )
    
    @classmethod
    def sample(cls, state: BufferState, num_elements: int, key: chex.PRNGKey) -> list[Any]:
        deque = state.deque
        max_idx = lax.select(state.done, state.capacity, state.ptr)
        idxs = jr.permutation(key, max_idx)[:num_elements]

        return [deque[i] for i in idxs]

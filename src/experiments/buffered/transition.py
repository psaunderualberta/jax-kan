import equinox as eqx
import chex


class Transition(eqx.Module):
    obs: chex.Array
    action: int
    next_obs: chex.Array
    done: bool
        
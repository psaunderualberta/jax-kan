import equinox as eqx
from optax import softmax_cross_entropy
from jax import vmap, numpy as jnp, random as jr
from jax import lax
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


def q_epsilon_greedy(q_network, state, epsilon: float, key: chex.PRNGKey) -> tuple[int, float, bool]:
    """Select an action using epsilon-greedy policy."""
    key, eps_key, action_key = jr.split(key, 3)

    q_values = q_network(state)
    explore = jr.uniform(eps_key) < epsilon
    greedy_action = jnp.argmax(q_values, axis=-1).squeeze()
    action = lax.select(
        explore,
        jr.randint(action_key, (), 0, q_network.num_actions()),
        greedy_action
    )

    q_value = q_values[action]
    explored = action != greedy_action
    return action, q_value, explored


def q_td_error(model, state, action, reward, done, next_state, gamma):
    q = model(state)[:, action]
    q_prime = jnp.argmax(model(next_state), axis=1)

    return (
        reward
        + (1 - done) * gamma * lax.stop_gradient(q_prime)
        - q
    )



def q_huber_loss(model, state, action, reward, done, next_state, gamma):
    td_error = q_td_error(model, state, action, reward, done, next_state, gamma)
    td_error = td_error.mean()
    return lax.select(
        jnp.abs(td_error) <= 1,
        jnp.abs(td_error) - 0.5,
        0.5 * td_error ** 2
    )
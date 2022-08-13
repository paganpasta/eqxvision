import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


class DropPath(eqx.Module):
    """Effectively dropping a sample from the call.
    Often used inside a network along side a residual connection.
    Equivalent to `torchvision.stochastic_depth`."""

    p: float
    inference: bool
    mode: str

    def __init__(self, p: float = 0.0, inference: bool = False, mode="atomic"):
        """**Arguments:**

        - `p`: The probability to drop a sample entirely during forward pass
        - `inference`: Defaults to `False`. If `True`, then the input is returned unchanged
        This may be toggled with `equinox.tree_inference`
        - `mode`: Can be set to `atomic` or `per_channel`. When `atomic`, the whole input is dropped or kept.
                If `per_channel`, then the decision on each channel is computed independently. Defaults to `atomic`
        """
        self.p = p
        self.inference = inference
        self.mode = mode

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to drop
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        """
        if self.inference or self.p == 0.0:
            return x
        if key is None:
            raise RuntimeError(
                "DropPath requires a key when running in non-deterministic mode. Did you mean to enable inference?"
            )

        keep_prob = 1 - self.p
        if self.mode == "atomic":
            return x * jrandom.bernoulli(key, p=keep_prob)
        else:
            return x * jnp.expand_dims(
                jrandom.bernoulli(key, p=keep_prob, shape=[x.shape[0]]).reshape(-1),
                axis=[i for i in range(1, len(x.shape))],
            )

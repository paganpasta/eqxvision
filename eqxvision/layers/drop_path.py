import equinox as eqx
import jax
import jax.random as jrandom
from equinox.custom_types import Array


class DropPath(eqx.Module):
    """Effectively dropping a sample from the call.
    Often used inside a network along side a residual connection."""

    p: float
    inference: bool

    def __init__(self, p: float = 0.0, inference: bool = False):
        """**Arguments:**

        - `p`: The probability to drop a sample entirely during forward pass.
        - `inference`: Defaults to `False`. If `True`, then the input is returned unchanged.
        This may be toggled with `equinox.tree_inference`.
        """
        self.p = p
        self.inference = inference

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to drop.
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
        return x * jrandom.bernoulli(key, p=keep_prob)

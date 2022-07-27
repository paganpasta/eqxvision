from typing import Any, Callable, Optional

import equinox as eqx
import jax


class Activation(eqx.Module):
    """Wrapper around Callables to make them eqx.Modules. Useful for `nn.Sequential`."""

    activation: Callable

    def __init__(
        self,
        activation: Callable,
    ):
        """**Arguments:**

        - `activation`: The `callable` to be wrapped in `equinox.Module`.
        """
        self.activation = activation

    def __call__(self, x: Any, *, key: Optional["jax.random.PRNGKey"] = None) -> Any:
        """**Arguments:**

        - `x`: A JAX `ndarray`.
        - `key`: Ignored.
        **Returns:**

        The output of the activation function.
        """
        return self.activation(x)

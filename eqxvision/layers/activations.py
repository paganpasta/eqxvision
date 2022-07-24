from typing import Optional
from equinox.custom_types import Array

import equinox as eqx
import jax
import jax.nn as jnn


class ReLU(eqx.Module):
    def __call__(
            self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of any shape.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of the same shape with  elementwise relu applied.
        """

        return jnn.relu(x)

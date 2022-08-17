from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.custom_types import Array


class LayerNorm2d(eqx.nn.LayerNorm):
    """Wraps `eqx.nn.LayerNorm` for an easy to apply channelwise-variant."""

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array of shape `(channels, dim_0, dim_1)`
        - `key`: Ignored

        **Returns:**

        Output of `eqx.nn.LayerNorm` applied to each `dim_0*dim_1 x c` entry.
        """
        c, h, w = x.shape
        x = jnp.transpose(x.reshape(c, -1))
        x = jax.vmap(super(LayerNorm2d, self).__call__)(x)
        x = jnp.transpose(x).reshape(c, h, w)
        return x


class Linear2d(eqx.nn.Linear):
    """Wraps `eqx.nn.Linear` for an easy to apply channelwise-variant."""

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array of shape `(channels, dim_0, dim_1)`
        - `key`: Ignored

        **Returns:**

        Output of `eqx.nn.Linear` applied to each `dim_0*dim_1 x c` entry.
        """
        c, h, w = x.shape
        x = jnp.transpose(x.reshape(c, -1))
        x = jax.vmap(super(Linear2d, self).__call__)(x)
        x = jnp.transpose(x).reshape(self.out_features, h, w)
        return x

from typing import Callable, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as jrandom
from equinox.custom_types import Array


class SqueezeExcitation(eqx.Module):
    """A simple port of `torchvision.ops.misc.SqueezeExcitation`"""

    avgpool: nn.AdaptiveAvgPool2d
    fc1: nn.Conv2d
    fc2: nn.Conv2d
    activation: nn.Lambda
    scale_activation: nn.Lambda

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable = jnn.relu,
        scale_activation: Callable = jnn.sigmoid,
        *,
        key: "jax.random.PRNGKey" = None
    ) -> None:
        """**Arguments:**

        - `input_channels`: Number of channels in the input image
        - `squeeze_channels`: Number of squeeze channels
        - `activation`: ``delta`` activation. Defaults to `jax.nn.relu`
        - `scale_activation`: `sigma`` activation. Defaults to `jax.nn.sigmoid`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        keys = jrandom.split(key, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1, key=keys[0])
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1, key=keys[1])
        self.activation = nn.Lambda(activation)
        self.scale_activation = nn.Lambda(scale_activation)

    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Forwarded to individual `eqx.Module` attributes
        """
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale, key=key)
        scale = self.fc2(scale)
        return x * self.scale_activation(scale, key=key)

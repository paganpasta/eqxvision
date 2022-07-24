from typing import Any, Optional
from equinox.custom_types import Array

import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn

from ...layers import ReLU

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


class AlexNet(eqx.Module):
    """A simple port of torchvision.models.alexnet"""
    features: eqx.Module
    avgpool: eqx.Module
    classifier: eqx.Module

    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.5,
            *,
            key: Optional["jax.random.PRNGKey"] = jrandom.PRNGKey(0)
    ) -> None:
        """**Arguments:**

        - `num_classes`: Number of classes. Decides the shape of the final output `(num_classes,)`.
        - `dropout`: Parameter used for the `equinox.nn.Dropout` layers.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        keys = jrandom.split(key, 21)

        self.features = eqx.nn.Sequential(
            [
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, key=keys[0]),
                ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2, key=keys[3]),
                ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1, key=keys[6]),
                ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1, key=keys[8]),
                ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, key=keys[10]),
                ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
             ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096, key=keys[15]),
                ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096, key=keys[18]),
                ReLU(),
                nn.Linear(4096, num_classes, key=keys[20])
            ]
        )

    def __call__(
        self, x: Array,
        *,
        key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

            - `x`: The input. Should be a JAX array with `3` channels.
            - `key`: Utilised by few layers in the network such as `nn.Dropout`.
        """
        keys = jrandom.split(key, 3)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x, key=keys[1])
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[2])
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    """
    model = AlexNet(**kwargs)
    #TODO: Add comptability with torch weights
    return model

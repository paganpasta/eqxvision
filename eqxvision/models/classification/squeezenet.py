from typing import Any, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array

from ...utils import load_torch_weights, MODEL_URLS


class Fire(eqx.Module):
    inplanes: int
    squeeze: nn.Conv2d
    squeeze_activation: nn.Lambda
    expand1x1: nn.Conv2d
    expand1x1_activation: nn.Lambda
    expand3x3: nn.Conv2d
    expand3x3_activation: nn.Lambda

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        key=None,
    ) -> None:
        super().__init__()
        keys = jrandom.split(key, 3)
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, key=keys[0])
        self.squeeze_activation = nn.Lambda(jnn.relu)
        self.expand1x1 = nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1, key=keys[1]
        )
        self.expand1x1_activation = nn.Lambda(jnn.relu)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1, key=keys[2]
        )
        self.expand3x3_activation = nn.Lambda(jnn.relu)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        x = self.squeeze_activation(self.squeeze(x))
        return jnp.concatenate(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            axis=0,
        )


class SqueezeNet(eqx.Module):
    """A simple port of `torchvision.models.squeezenet`"""

    features: nn.Sequential
    classifier: nn.Sequential

    def __init__(
        self,
        version: str = "1_0",
        num_classes: int = 1000,
        dropout: float = 0.5,
        *,
        key: Optional["jax.random.PRNGKey"] = None
    ) -> None:
        """**Arguments:**

        - `version`: Specifies the version of the network. Defaults to `1_0`
        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `dropout`: The probability parameter for `equinox.nn.Dropout`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 10)
        if version == "1_0":
            self.features = nn.Sequential(
                [
                    nn.Conv2d(3, 96, kernel_size=7, stride=2, key=keys[0]),
                    nn.Lambda(jnn.relu),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(96, 16, 64, 64, key=keys[1]),
                    Fire(128, 16, 64, 64, key=keys[2]),
                    Fire(128, 32, 128, 128, key=keys[3]),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(256, 32, 128, 128, key=keys[4]),
                    Fire(256, 48, 192, 192, key=keys[5]),
                    Fire(384, 48, 192, 192, key=keys[6]),
                    Fire(384, 64, 256, 256, key=keys[7]),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(512, 64, 256, 256, key=keys[8]),
                ]
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                [
                    nn.Conv2d(3, 64, kernel_size=3, stride=2, key=keys[0]),
                    nn.Lambda(jnn.relu),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(64, 16, 64, 64, key=keys[1]),
                    Fire(128, 16, 64, 64, key=keys[2]),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(128, 32, 128, 128, key=keys[3]),
                    Fire(256, 32, 128, 128, key=keys[4]),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    Fire(256, 48, 192, 192, key=keys[5]),
                    Fire(384, 48, 192, 192, key=keys[6]),
                    Fire(384, 64, 256, 256, key=keys[7]),
                    Fire(512, 64, 256, 256, key=keys[8]),
                ]
            )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1, key=keys[9])
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                final_conv,
                nn.Lambda(jnn.relu),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        x = self.features(x)
        x = self.classifier(x, key=key)
        return jnp.ravel(x)


def _squeezenet(version: str, pretrained: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = "squeezenet" + version
        model = load_torch_weights(model, url=MODEL_URLS[arch])
    return model


def squeezenet1_0(pretrained: bool = False, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.

    **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint
    """
    return _squeezenet("1_0", pretrained, **kwargs)


def squeezenet1_1(pretrained: bool = False, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.

    **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint
    """
    return _squeezenet("1_1", pretrained, **kwargs)

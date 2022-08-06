from typing import Any, Callable, List, Optional

import equinox as eqx
import equinox.experimental as eqxex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array

from ...layers import ConvNormActivation
from ...utils import _make_divisible


class InvertedResidual(eqx.Module):
    stride: int
    use_res_connect: int
    conv: nn.Sequential
    out_channels: int

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()

        keys = jrandom.split(key, 3)

        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = eqxex.BatchNorm

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[eqx.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=jnn.relu,
                    key=keys[0],
                )
            )
        layers.extend(
            [
                # dw
                ConvNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=jnn.relu,
                    key=keys[1],
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, use_bias=False, key=keys[2]),
                norm_layer(oup, axis_name="batch"),
            ]
        )
        self.conv = nn.Sequential(layers)
        self.out_channels = oup

    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Forwarded to individual `eqx.Module` attributes
        """
        if self.use_res_connect:
            return x + self.conv(x, key=key)
        else:
            return self.conv(x, key=key)


class MobileNetV2(eqx.Module):
    """A simple port of `torchvision.models.mobilenetv2`"""

    features: nn.Sequential
    classifier: nn.Linear
    pool: nn.AdaptivePool

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., eqx.Module]] = None,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        dropout: float = 0.2,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> None:
        """**Arguments:**

        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `width_mult`: Adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
        - `inverted_residual_setting`: Network structure
        - `round_nearest`: Round the number of channels in each layer to be a multiple of this number
            Set to `1` to turn off rounding
        - `block`: Module specifying inverted residual building block for mobilenet
        - `norm_layer`: Module specifying the normalization layer to use
        - `dropout`: The dropout probability
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()

        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 2)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = eqxex.BatchNorm

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: List[eqx.Module] = [
            ConvNormActivation(
                3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=jnn.relu,
                key=keys[0],
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                keys = jrandom.split(keys[1], 2)
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        key=keys[0],
                    )
                )
                input_channel = output_channel
        # building last several layers
        keys = jrandom.split(keys[1], 2)
        features.append(
            ConvNormActivation(
                input_channel,
                last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=jnn.relu,
                key=keys[0],
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(features)

        # building classifier
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                nn.Linear(last_channel, num_classes, key=keys[1]),
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 3)
        x = self.features(x, key=keys[0])
        x = self.pool(x, key=keys[1])
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[2])
        return x


def mobilenet_v2(**kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    """
    model = MobileNetV2(**kwargs)
    return model

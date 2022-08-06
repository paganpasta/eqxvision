from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import equinox as eqx
import equinox.experimental as eqxex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array

from ...layers import ConvNormActivation
from ...layers import SqueezeExcitation as SElayer
from ...utils import _make_divisible


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(eqx.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    use_res_connect: int
    block: nn.Sequential
    out_channels: int

    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., eqx.Module],
        se_layer: Callable[..., eqx.Module] = partial(
            SElayer, scale_activation=jnn.hard_sigmoid
        ),
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        keys = jrandom.split(key, 4)
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[eqx.Module] = []
        activation_layer = jnn.hard_sigmoid if cnf.use_hs else jnn.relu

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    key=keys[0],
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                key=keys[1],
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(
                se_layer(cnf.expanded_channels, squeeze_channels, key=keys[2])
            )

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
                key=keys[3],
            )
        )

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_channels

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        result = self.block(x, key=key)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(eqx.Module):
    """A simple port of `torchvision.models.mobilenetv3`"""

    features: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    classifier: nn.Sequential

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., eqx.Module]] = None,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        dropout: float = 0.2,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> None:
        """**Arguments:**

        - `inverted_residual_setting`: Network structure
        - `last_channel`: The number of channels on the penultimate layer
        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `block`: Module specifying inverted residual building block for mobilenet
        - `norm_layer`: Module specifying the normalization layer to use
        - `dropout`: The dropout probability
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)

        keys = jrandom.split(key, 5)
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(eqxex.BatchNorm, eps=0.001, momentum=0.01)

        layers: List[eqx.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=jnn.hard_swish,
                key=keys[0],
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer, key=keys[1]))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=jnn.hard_swish,
                key=keys[2],
            )
        )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            [
                nn.Linear(lastconv_output_channels, last_channel, key=keys[3]),
                nn.Lambda(jnn.hard_swish),
                nn.Dropout(p=dropout),
                nn.Linear(last_channel, num_classes, key=keys[4]),
            ]
        )

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 3)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x, key=keys[1])
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[2])
        return x


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs: Any,
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(
                112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation
            ),  # C4
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    **kwargs: Any,
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    return model


def mobilenet_v3_large(**kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, **kwargs)


def mobilenet_v3_small(**kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, **kwargs)

import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.custom_types import Array

from ...layers import ConvNormActivation, DropPath, SqueezeExcitation
from ...utils import _make_divisible, load_torch_weights


@dataclass
class _MBConvConfigData:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., eqx.Module]

    @staticmethod
    def adjust_channels(
        channels: int, width_mult: float, min_value: Optional[int] = None
    ) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class _MBConvConfig(_MBConvConfigData):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., eqx.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = _MBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class _FusedMBConvConfig(_MBConvConfigData):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., eqx.Module]] = None,
    ) -> None:
        if block is None:
            block = _FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class _MBConv(eqx.Module):
    use_res_connect: bool
    block: nn.Sequential
    stochastic_depth: DropPath
    out_channels: int

    def __init__(
        self,
        cnf: _MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., eqx.Module],
        se_layer: Callable[..., eqx.Module] = SqueezeExcitation,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, 4)
        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[eqx.Module] = []
        activation_layer = jnn.silu

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    key=keys[0],
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                key=keys[1],
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=activation_layer,
                key=keys[2],
            )
        )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
                key=keys[3],
            )
        )

        self.block = nn.Sequential(layers)
        self.stochastic_depth = DropPath(stochastic_depth_prob, mode="per_channel")
        self.out_channels = cnf.out_channels

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        keys = jr.split(key, 2)
        result = self.block(x, key=keys[0])
        if self.use_res_connect:
            result = self.stochastic_depth(result, key=keys[1])
            result += x
        return result


class _FusedMBConv(eqx.Module):
    use_res_connect: bool
    block: nn.Sequential
    stochastic_depth: DropPath
    out_channels: int

    def __init__(
        self,
        cnf: _FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., eqx.Module],
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")
        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, 3)
        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[eqx.Module] = []
        activation_layer = jnn.silu

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    key=keys[0],
                )
            )

            # project
            layers.append(
                ConvNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                    key=keys[1],
                )
            )
        else:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    key=keys[2],
                )
            )

        self.block = nn.Sequential(layers)
        self.stochastic_depth = DropPath(stochastic_depth_prob, mode="local")
        self.out_channels = cnf.out_channels

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        keys = jr.split(key, 2)
        result = self.block(x, key=keys[0])
        if self.use_res_connect:
            result = self.stochastic_depth(result, key=keys[1])
            result += x
        return result


class EfficientNet(eqx.Module):
    """A simple port of `torchvision.models.efficientnet`."""

    features: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    classifier: nn.Sequential

    def __init__(
        self,
        inverted_residual_setting: Sequence[
            Union["_MBConvConfig", "_FusedMBConvConfig"]
        ],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional["eqx.Module"] = None,
        last_channel: Optional[int] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        """**Arguments:**

        - `inverted_residual_setting`: Network structure
        - `dropout`: The dropout probability
        - `stochastic_depth_prob`: Probability of dropping a sample along channels
        - `num_classes`: Number of classes in the classification task.
                         Also controls the final output shape `(num_classes,)`
        - `norm_layer`: Normalisation applied to the intermediate outputs
        - `last_channel`: The number of channels on the penultimate layer
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(
                [isinstance(s, _MBConvConfigData) for s in inverted_residual_setting]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        if norm_layer is None:
            norm_layer = eqx.experimental.BatchNorm

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
                activation_layer=jnn.silu,
                key=keys[0],
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[eqx.Module] = []
            for _ in range(cnf.num_layers):
                keys = jr.split(keys[1], 2)
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(
                    block_cnf.block(block_cnf, sd_prob, norm_layer, key=keys[0])
                )
                stage_block_id += 1

            layers.append(nn.Sequential(stage))

        keys = jr.split(keys[1], 2)
        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=jnn.silu,
                key=keys[0],
            )
        )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                nn.Linear(lastconv_output_channels, num_classes, key=keys[1]),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        keys = jr.split(key, 2)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[1])
        return x


def _efficientnet(
    arch: str,
    inverted_residual_setting: Sequence[Union[_MBConvConfig, _FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    torch_weights: str,
    **kwargs: Any,
) -> EfficientNet:
    model = EfficientNet(
        inverted_residual_setting, dropout, last_channel=last_channel, **kwargs
    )
    if torch_weights:
        model = load_torch_weights(model, torch_weights=torch_weights)

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[_MBConvConfig, _FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[_MBConvConfig, _FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            _MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            _FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            _FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            _FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            _MBConvConfig(4, 3, 2, 64, 128, 6),
            _MBConvConfig(6, 3, 1, 128, 160, 9),
            _MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            _FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            _FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            _FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            _MBConvConfig(4, 3, 2, 80, 160, 7),
            _MBConvConfig(6, 3, 1, 160, 176, 14),
            _MBConvConfig(6, 3, 2, 176, 304, 18),
            _MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            _FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            _FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            _FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            _MBConvConfig(4, 3, 2, 96, 192, 10),
            _MBConvConfig(6, 3, 1, 192, 224, 19),
            _MBConvConfig(6, 3, 2, 224, 384, 25),
            _MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B0 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )
    return _efficientnet(
        "efficientnet_b0",
        inverted_residual_setting,
        0.2,
        last_channel,
        torch_weights,
        **kwargs,
    )


def efficientnet_b1(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B1 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )
    return _efficientnet(
        "efficientnet_b1",
        inverted_residual_setting,
        0.2,
        last_channel,
        torch_weights,
        **kwargs,
    )


def efficientnet_b2(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B2 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )
    return _efficientnet(
        "efficientnet_b2",
        inverted_residual_setting,
        0.3,
        last_channel,
        torch_weights,
        **kwargs,
    )


def efficientnet_b3(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B3 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )
    return _efficientnet(
        "efficientnet_b3",
        inverted_residual_setting,
        0.3,
        last_channel,
        torch_weights,
        **kwargs,
    )


def efficientnet_b4(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B4 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )
    return _efficientnet(
        "efficientnet_b4",
        inverted_residual_setting,
        0.4,
        last_channel,
        torch_weights,
        **kwargs,
    )


def efficientnet_b5(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B5 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )
    return _efficientnet(
        "efficientnet_b5",
        inverted_residual_setting,
        0.4,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B6 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )
    return _efficientnet(
        "efficientnet_b6",
        inverted_residual_setting,
        0.5,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """EfficientNet B7 model architecture from the [EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks](https://arxiv.org/abs/1905.11946) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )
    return _efficientnet(
        "efficientnet_b7",
        inverted_residual_setting,
        0.5,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_v2_s(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        "efficientnet_v2_s",
        inverted_residual_setting,
        0.2,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_m(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        "efficnetnet_v2_m",
        inverted_residual_setting,
        0.3,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_l(torch_weights: str = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        "efficientnet_v2_l",
        inverted_residual_setting,
        0.4,
        last_channel,
        torch_weights,
        norm_layer=partial(eqx.experimental.BatchNorm, eps=1e-03),
        **kwargs,
    )

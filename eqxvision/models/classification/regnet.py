from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.custom_types import Array

from ...layers import ConvNormActivation, SqueezeExcitation
from ...utils import _make_divisible, load_torch_weights


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Optional[Callable],
        activation_layer: Optional[Callable],
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            key=key,
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Optional[Callable],
        activation_layer: Optional[Callable],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        *,
        key: "jax.random.PRNGKey",
    ) -> None:
        layers = []
        keys = jr.split(key, 4)
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers.append(
            ConvNormActivation(
                width_in,
                w_b,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                key=keys[0],
            )
        )
        layers.append(
            ConvNormActivation(
                w_b,
                w_b,
                kernel_size=3,
                stride=stride,
                groups=g,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                key=keys[1],
            )
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers.append(
                SqueezeExcitation(
                    input_channels=w_b,
                    squeeze_channels=width_se_out,
                    activation=activation_layer,
                    key=keys[2],
                )
            )

        layers.append(
            ConvNormActivation(
                w_b,
                width_out,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=None,
                key=keys[3],
            )
        )
        super().__init__(layers)


class ResBottleneckBlock(eqx.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    proj: eqx.Module
    f: eqx.Module
    activation: Callable

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Optional[Callable],
        activation_layer: Optional[Callable],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = nn.Identity()
        should_proj = (width_in != width_out) or (stride != 1)
        keys = jr.split(key, 2)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
                key=keys[0],
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
            key=keys[1],
        )
        self.activation = activation_layer

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        keys = jr.split(key, 2)
        x = self.proj(x, key=keys[0]) + self.f(x, key=keys[1])
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: eqx.Module,
        norm_layer: Callable,
        activation_layer: Callable,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        keys = jr.split(key, depth)
        blocks = []
        for i in range(depth):
            blocks.append(
                block_constructor(
                    width_in if i == 0 else width_out,
                    width_out,
                    stride if i == 0 else 1,
                    norm_layer,
                    activation_layer,
                    group_width,
                    bottleneck_multiplier,
                    se_ratio,
                    key=keys[i],
                )
            )

        super().__init__(blocks)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.
        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space
        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.
        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = jnp.arange(depth) * w_a + w_0
        block_capacity = jnp.round(jnp.log(widths_cont / w_0) / jnp.log(w_m))
        block_widths = (
            (jnp.round(jnp.divide(w_0 * jnp.power(w_m, block_capacity), QUANT)) * QUANT)
            .astype(jnp.int32)
            .tolist()
        )
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            jnp.diff(jnp.asarray([d for d, t in enumerate(splits) if t]))
            .astype(jnp.int32)
            .tolist()
        )

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [
            _make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)
        ]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(eqx.Module):
    """A simple port of `torchvision.models.regnet`"""

    stem: eqx.Module
    trunk_output: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    fc: eqx.Module

    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[eqx.Module] = None,
        block_type: Optional[eqx.Module] = None,
        norm_layer: Optional[eqx.Module] = None,
        activation: Optional[Callable] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        """**Arguments:**

        - `block_params`: Configuration for the building blocks of the network
        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `stem_width`: Width of stems in the model
        - `stem_type`: Block type for the stems
        - `block_type`: Type of block to be used in building the model
        - `norm_layer`: Normalisation to be applied on the inputs. Defaults to `BatchNorm`
        - `activation`: Activation to be applied to the intermediate outputs. Defaults to `jax.nn.relu`
        - `key`:         - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """

        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = eqx.experimental.BatchNorm
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = jnn.relu

        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, 2)
        # Ad hoc stem
        self.stem = stem_type(
            3, stem_width, norm_layer, activation, key=keys[0]  # width_in
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            keys = jr.split(keys[1], 2)
            blocks.append(
                AnyStage(
                    current_width,
                    width_out,
                    stride,
                    depth,
                    block_type,
                    norm_layer,
                    activation,
                    group_width,
                    bottleneck_multiplier,
                    block_params.se_ratio,
                    key=keys[0],
                ),
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=current_width, out_features=num_classes, key=keys[1]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jr.split(key, 2)
        x = self.stem(x, key=keys[0])
        x = self.trunk_output(x, key=keys[1])
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x)
        return x


def _regnet(
    arch: str,
    block_params: BlockParams,
    torch_weights: str,
    **kwargs: Any,
) -> RegNet:

    norm_layer = kwargs.pop(
        "norm_layer", partial(eqx.experimental.BatchNorm, eps=1e-05, momentum=0.1)
    )
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    if torch_weights:
        model = load_torch_weights(model, torch_weights=torch_weights)
    return model


def regnet_y_400mf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """Constructs a RegNetY_400MF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25
    )
    return _regnet(
        arch="regnet_y_400mf",
        block_params=params,
        torch_weights=torch_weights,
        **kwargs,
    )


def regnet_y_800mf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25
    )
    return _regnet("regnet_y_800mf", params, torch_weights, **kwargs)


def regnet_y_1_6gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_1.6GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25
    )
    return _regnet("regnet_y_1_6gf", params, torch_weights, **kwargs)


def regnet_y_3_2gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_3.2GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25
    )
    return _regnet("regnet_y_3_2gf", params, torch_weights, **kwargs)


def regnet_y_8gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_8GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25
    )
    return _regnet("regnet_y_8gf", params, torch_weights, **kwargs)


def regnet_y_16gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_16GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25
    )
    return _regnet("regnet_y_16gf", params, torch_weights, **kwargs)


def regnet_y_32gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25
    )
    return _regnet("regnet_y_32gf", params, torch_weights, **kwargs)


def regnet_y_128gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_128GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25
    )
    return _regnet("regnet_y_128gf", params, torch_weights, **kwargs)


def regnet_x_400mf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16
    )
    return _regnet("regnet_x_400mf", params, torch_weights, **kwargs)


def regnet_x_800mf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16
    )
    return _regnet("regnet_x_800mf", params, torch_weights, **kwargs)


def regnet_x_1_6gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24
    )
    return _regnet("regnet_x_1_6gf", params, torch_weights, **kwargs)


def regnet_x_3_2gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48
    )
    return _regnet("regnet_x_3_2gf", params, torch_weights, **kwargs)


def regnet_x_8gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_8GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120
    )
    return _regnet("regnet_x_8gf", params, torch_weights, **kwargs)


def regnet_x_16gf(torch_weights: str = None, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_16GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    params = BlockParams.from_init_params(
        depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128
    )
    return _regnet("regnet_x_16gf", params, torch_weights, **kwargs)


def regnet_x_32gf(torch_weights: bool = False, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_32GF architecture from
    [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    params = BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168
    )
    return _regnet("regnet_x_32gf", params, torch_weights, **kwargs)

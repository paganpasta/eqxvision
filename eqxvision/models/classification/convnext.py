from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array

from ...layers import ConvNormActivation, DropPath, LayerNorm2d, Linear2d
from ...utils import CLASSIFICATION_URLS, load_torch_weights


class CNBlock(eqx.Module):
    layer_scale: jnp.ndarray
    block: nn.Sequential
    stochastic_depth: DropPath

    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., eqx.Module]] = LayerNorm2d,
        *,
        key=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        keys = jrandom.split(key, 4)
        self.block = nn.Sequential(
            [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=7,
                    padding=3,
                    groups=dim,
                    use_bias=True,
                    key=keys[0],
                ),
                norm_layer(dim),
                Linear2d(
                    in_features=dim, out_features=4 * dim, use_bias=True, key=keys[1]
                ),
                nn.Lambda(jnn.gelu),
                Linear2d(
                    in_features=4 * dim, out_features=dim, use_bias=True, key=keys[2]
                ),
            ]
        )
        self.layer_scale = jnp.asarray(
            jnp.ones(shape=(dim, 1, 1)) * layer_scale, dtype=jnp.float32
        )
        self.stochastic_depth = DropPath(p=stochastic_depth_prob, mode="local")

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Utilised by few layers in the network such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 2)
        result = self.layer_scale * self.block(x, key=keys[0])
        result = self.stochastic_depth(result, key=keys[1])
        result += x
        return result


class _CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(eqx.Module):
    """A simple port of `torchvision.models.convnext`."""

    features: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    classifier: nn.Sequential

    def __init__(
        self,
        block_setting: Sequence["_CNBlockConfig"],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional["eqx.Module"] = None,
        norm_layer: Optional["eqx.Module"] = None,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        """

        - `block_setting`: Configuration of the computational blocks
        - `stochastic_depth_prob`: Probability of dropping a sample along channels
        - `layer_scale`: Scale applied to the output of computational stem
        - `num_classes`: Number of classes in the classification task.
                         Also controls the final output shape `(num_classes,)`
        - `block`: The block type used within the network
        - `norm_layer`: Normalisation applied to the intermediate outputs
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, _CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 2)

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[eqx.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels=3,
                out_channels=firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                use_bias=True,
                key=keys[0],
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[eqx.Module] = []
            for _ in range(cnf.num_layers):
                keys = jrandom.split(keys[1], 2)
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(
                    block(cnf.input_channels, layer_scale, sd_prob, key=keys[0])
                )
                stage_block_id += 1
            layers.append(nn.Sequential(stage))
            if cnf.out_channels is not None:
                keys = jrandom.split(keys[1], 2)
                # Downsampling
                layers.append(
                    nn.Sequential(
                        [
                            norm_layer(cnf.input_channels),
                            nn.Conv2d(
                                cnf.input_channels,
                                cnf.out_channels,
                                kernel_size=2,
                                stride=2,
                                key=keys[0],
                            ),
                        ]
                    )
                )

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            [
                norm_layer(lastconv_output_channels),
                nn.Lambda(jnp.ravel),
                nn.Linear(lastconv_output_channels, num_classes, key=keys[1]),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        x = self.features(x, key=key)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def _convnext(
    arch: str,
    block_setting: List[_CNBlockConfig],
    stochastic_depth_prob: float,
    torch_weights: str,
    **kwargs: Any,
) -> ConvNeXt:
    model = ConvNeXt(
        block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs
    )
    if torch_weights:
        if arch not in CLASSIFICATION_URLS:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        model = load_torch_weights(model, torch_weights=CLASSIFICATION_URLS[arch])
    return model


def convnext_tiny(*, torch_weights: str = None, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Tiny model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    block_setting = [
        _CNBlockConfig(96, 192, 3),
        _CNBlockConfig(192, 384, 3),
        _CNBlockConfig(384, 768, 9),
        _CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(
        "convnext_tiny",
        block_setting=block_setting,
        stochastic_depth_prob=stochastic_depth_prob,
        torch_weights=torch_weights,
        **kwargs,
    )


def convnext_small(*, torch_weights: str = None, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Small model architecture from the
    [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    block_setting = [
        _CNBlockConfig(96, 192, 3),
        _CNBlockConfig(192, 384, 3),
        _CNBlockConfig(384, 768, 27),
        _CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(
        "convnext_small", block_setting, stochastic_depth_prob, torch_weights, **kwargs
    )


def convnext_base(*, torch_weights: str = None, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Base model architecture from the
    [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    block_setting = [
        _CNBlockConfig(128, 256, 3),
        _CNBlockConfig(256, 512, 3),
        _CNBlockConfig(512, 1024, 27),
        _CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(
        "convnext_base", block_setting, stochastic_depth_prob, torch_weights, **kwargs
    )


def convnext_large(*, torch_weights: str = None, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Large model architecture from the
    [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) paper.

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    block_setting = [
        _CNBlockConfig(192, 384, 3),
        _CNBlockConfig(384, 768, 3),
        _CNBlockConfig(768, 1536, 27),
        _CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(
        "convnext_large", block_setting, stochastic_depth_prob, torch_weights, **kwargs
    )

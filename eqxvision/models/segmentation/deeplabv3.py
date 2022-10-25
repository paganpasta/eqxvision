from typing import Callable, List, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ...experimental import intermediate_layer_getter
from ...utils import load_torch_weights
from ..classification import resnet
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead


class DeepLabV3(_SimpleSegmentationModel):
    """Ported from `torchvision.models.segmentation.deeplabv3`"""

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, key=None) -> None:
        keys = jr.split(key, 3)
        super().__init__(
            [
                ASPP(in_channels, [12, 24, 36], key=keys[0]),
                nn.Conv2d(256, 256, 3, padding=1, use_bias=False, key=keys[1]),
                eqx.experimental.BatchNorm(256, axis_name="batch"),
                nn.Lambda(jnn.relu),
                nn.Conv2d(256, out_channels, 1, key=keys[2]),
            ]
        )


class ASPPConv(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, key=None
    ) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                use_bias=False,
                key=key,
            ),
            eqx.experimental.BatchNorm(out_channels, axis_name="batch"),
            nn.Lambda(jnn.relu),
        ]
        super().__init__(modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, key=None) -> None:
        super().__init__(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, use_bias=False, key=key),
                eqx.experimental.BatchNorm(out_channels, axis_name="batch"),
                nn.Lambda(jnn.relu),
            ]
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        size = x.shape
        x = super().__call__(x)
        return jax.image.resize(x, x.shape[:-2] + size[-2:], method="bilinear")


class ASPP(eqx.Module):
    convs: eqx.Module
    project: eqx.Module

    def __init__(
        self,
        in_channels: int,
        atrous_rates: List[int],
        out_channels: int = 256,
        key=None,
    ) -> None:
        super().__init__()

        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, len(atrous_rates) + 3)
        modules = []
        modules.append(
            nn.Sequential(
                [
                    nn.Conv2d(
                        in_channels, out_channels, 1, use_bias=False, key=keys[0]
                    ),
                    eqx.experimental.BatchNorm(out_channels, axis_name="batch"),
                    nn.Lambda(jnn.relu),
                ]
            )
        )
        rates = tuple(atrous_rates)
        for i, rate in enumerate(rates):
            modules.append(ASPPConv(in_channels, out_channels, rate, key=keys[i + 1]))

        modules.append(ASPPPooling(in_channels, out_channels, key=keys[-2]))

        self.convs = nn.Sequential(modules)

        self.project = nn.Sequential(
            [
                nn.Conv2d(
                    len(self.convs) * out_channels,
                    out_channels,
                    1,
                    use_bias=False,
                    key=keys[-1],
                ),
                eqx.experimental.BatchNorm(out_channels, axis_name="batch"),
                nn.Lambda(jnn.relu),
                nn.Dropout(0.5),
            ]
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        _res = []
        for conv in self.convs.layers:
            _res.append(conv(x))
        x = jnp.concatenate(_res, axis=0)
        return self.project(x, key=key)


def deeplabv3(
    num_classes: Optional[int] = 21,
    backbone: "eqx.Module" = None,
    intermediate_layers: Callable = None,
    classifier_module: "eqx.Module" = None,
    classifier_in_channels: int = 2048,
    aux_classifier_module: "eqx.Module" = None,
    aux_in_channels: int = 1024,
    silence_layers: Callable = None,
    torch_weights: str = None,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
) -> DeepLabV3:
    """Implements DeepLabV3 model from
    [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) paper.

    !!! info "Sample call"
        ```python
        net = deeplabv3(
            backbone=resnet50(replace_stride_with_dilation=[False, True, True]),
            intermediate_layers=lambda x: [x.layer3, x.layer4],
            aux_in_channels=1024,
            torch_weights=SEGMENTATION_URLS["deeplabv3_resnet50"]
        )
        ```


    **Arguments:**

    - `num_classes`: Number of classes in the segmentation task.
                    Also controls the final output shape `(num_classes, height, width)`. Defaults to `21`
    - `backbone`: The neural network to use for extracting features. If `None`, then all params are set to
                `DeepLabV3_RESNET50` with `untrained` weights
    - `intermediate_layers`: Layers from `backbone` to be used for generating output maps. Default sets it to
        `layer3` and `layer4` from `DeepLabV3_RESNET50`
    - `classifier_module`: Uses the `DeepLabHead` by default
    - `classifier_in_channels`: Number of input channels from the last intermediate layer
    - `aux_classifier_module`: Uses the `FCNHead` by default
    - `aux_in_channels`: Number of channels in the auxiliary output. It is used when number of intermediate_layers
        is equal to 2.
    - `silence_layers`: Layers of a network not used in training. Typically, for a backbone ported from classification
        the `fc` layers can be dropped. This is particularly useful when loading weights from `torchvision`. By
        default, `.fc` layer of a model is set to identity to avoid tracking weights.
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
    """
    if key is None:
        key = jr.PRNGKey(0)
    keys = jr.split(key, 2)
    if not classifier_module:
        classifier_module = DeepLabHead
    if not aux_classifier_module:
        aux_classifier_module = FCNHead
    if backbone is None:
        backbone = resnet.resnet50(
            replace_stride_with_dilation=[False, True, True],
        )
    num_layers = len(intermediate_layers(backbone))

    if silence_layers is None:
        silence_layers = lambda x: x.fc
    if aux_in_channels is not None and num_layers != 2:
        raise ValueError(
            "aux_in_channels requires the intermediate_layers to return exactly 2 layers "
            "corresponding to aux and final."
        )
    if aux_in_channels is None and num_layers != 1:
        raise ValueError(
            f"With no aux_in_channels, the aux layer is disabled. Received {num_layers} "
            f"from intermediate_layers, expected number of layers is 1."
        )

    backbone = eqx.tree_at(silence_layers, backbone, replace_fn=lambda x: nn.Identity())
    backbone = intermediate_layer_getter(backbone, intermediate_layers)

    classifier = classifier_module(
        in_channels=classifier_in_channels, out_channels=num_classes, key=keys[0]
    )
    if aux_in_channels is not None:
        aux_classifier = aux_classifier_module(
            in_channels=aux_in_channels, out_channels=num_classes, key=keys[1]
        )
    else:
        aux_classifier = None
    model = DeepLabV3(backbone, classifier, aux_classifier)

    if torch_weights:
        return load_torch_weights(model, torch_weights=torch_weights)

    return model

from typing import Callable, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as jr

from ...utils import CLASSIFICATION_URLS, intermediate_layer_getter, load_torch_weights
from ..classification import resnet
from ._utils import _SimpleSegmentationModel


class FCN(_SimpleSegmentationModel):
    """Ported from `torchvision.models.segmentation.fcn`"""


class FCNHead(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, *, key: ["jax.random.PRNGKey"]
    ) -> None:
        keys = jr.split(key, 2)
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(
                in_channels, inter_channels, 3, padding=1, use_bias=False, key=keys[0]
            ),
            eqx.experimental.BatchNorm(inter_channels, axis_name="batch"),
            nn.Lambda(jnn.relu),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, key=keys[1]),
        ]
        super().__init__(layers)


def fcn(
    num_classes: Optional[int] = 21,
    backbone: "eqx.Module" = None,
    intermediate_layers: Callable = None,
    classifier_module: "eqx.Module" = None,
    classifier_in_channels: int = 2048,
    aux_in_channels: int = None,
    silence_layers: Callable = None,
    torch_weights: str = None,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
) -> FCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the [Fully Convolutional
    Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) paper.

    **Arguments:**

    - `num_classes`: Number of classes in the segmentation task.
                    Also controls the final output shape `(num_classes, height, width)`. Defaults to `21`
    - `backbone`: The neural network to use for extracting features. If `None`, then all params are set to
                `FCN_RESNET50` with a **pre-trained** backbone but an **untrained** FCN
    - `intermediate_layers`: Layers from `backbone` to be used for generating output maps. Default sets it to
        `layer3` and `layer4` from `FCN_RESNET50`
    - `classifier_module`: Uses the `FCNHead` by default
    - `classifier_in_channels`: Number of input channels from the last intermediate layer
    - `aux_in_channels`: Number of channels in the auxiliary output. It is used when number of intermediate_layers
        is equal to 2.
    - `silence_layers`: Layers of a network not used in training. Typically, for a backbone ported from classification
        the `fc` layers can be dropped. This is particularly useful when loading weights from `torchvision`. By
        default, fc layer of a model is set to identity to avoid tracking weights.
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    if key is None:
        key = jr.PRNGKey(0)
    keys = jr.split(key, 2)

    if backbone is None:
        backbone = resnet.resnet50(
            torch_weights=CLASSIFICATION_URLS["resnet50"],
            replace_stride_with_dilation=[False, True, True],
        )
    num_layers = len(intermediate_layers(backbone))

    if classifier_module is None:
        classifier_module = FCNHead
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
            f"from intermediate_layers, expected number of layers 1."
        )

    backbone = eqx.tree_at(silence_layers, backbone, replace_fn=lambda x: nn.Identity())
    backbone = intermediate_layer_getter(backbone, intermediate_layers)
    classifier = classifier_module(
        in_channels=classifier_in_channels, out_channels=num_classes, key=keys[0]
    )
    if aux_in_channels is not None:
        aux_classifier = classifier_module(
            in_channels=aux_in_channels, out_channels=num_classes, key=keys[1]
        )
    else:
        aux_classifier = None
    model = FCN(backbone, classifier, aux_classifier)

    if torch_weights:
        return load_torch_weights(model, torch_weights=torch_weights)

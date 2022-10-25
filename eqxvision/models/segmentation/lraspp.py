from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as jr
from jaxtyping import Array

from ...experimental import intermediate_layer_getter
from ...utils import load_torch_weights
from ..classification.mobilenetv3 import mobilenet_v3_large


class LRASPP(eqx.Module):
    """Implements a Lite R-ASPP Network for semantic segmentation from
    ["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244).
    """

    backbone: eqx.Module
    classifier: eqx.Module

    def __init__(
        self,
        backbone: "eqx.Module",
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> None:
        """**Arguments:**

        - `backbone`: the network used to compute the features for the model. The intermediate layers of the
        `backbone` should be wrapped for obtaining intermediate features

        - `low_channels`: the number of channels of the low level features

        - `high_channels`: the number of channels of the high level features

        - `num_classes`: number of output classes of the model (including the background)

        - `inter_channels`: the number of channels for intermediate computations

        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter

        """
        super().__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels, key=key
        )

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Tuple[Any, Array]:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`

        **Returns:**
        A tuple with outputs from the intermediate and last layers.
        """
        _, features = self.backbone(x)
        out = self.classifier(features)
        out = jax.image.resize(out, out.shape[:-2] + x.shape[-2:], method="bilinear")
        return None, out


class LRASPPHead(eqx.Module):
    cbr: eqx.Module
    scale: eqx.Module
    low_classifier: eqx.Module
    high_classifier: eqx.Module

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int,
        key=None,
    ) -> None:
        super().__init__()
        keys = jr.split(key, 4)
        self.cbr = nn.Sequential(
            [
                nn.Conv2d(
                    high_channels, inter_channels, 1, use_bias=False, key=keys[0]
                ),
                eqx.experimental.BatchNorm(inter_channels, axis_name="batch"),
                nn.Lambda(jnn.relu),
            ]
        )
        self.scale = nn.Sequential(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    high_channels, inter_channels, 1, use_bias=False, key=keys[1]
                ),
                nn.Lambda(jnn.sigmoid),
            ]
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1, key=keys[2])
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1, key=keys[3])

    def __call__(
        self, x: Tuple[Array], *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        low = x[0]
        high = x[1]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = jax.image.resize(x, x.shape[:-2] + low.shape[-2:], method="bilinear")

        return self.low_classifier(low) + self.high_classifier(x)


def lraspp_mobilenet_v3_large(
    num_classes: Optional[int] = 21,
    backbone: "eqx.Module" = None,
    intermediate_layers: Callable = None,
    torch_weights: str = None,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
) -> LRASPP:
    """Implements a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper.

    !!! info "Sample call"
        ```python
        net = lraspp_mobilenet_v3_large(
            backbone=mobilenet_v3_large(dilated=True),
            intermediate_layers=lambda x: [4, 16],
            torch_weights=SEGMENTATION_URLS['lraspp_mobilenetv3_large']
        )
        ```

    **Arguments:**

    - `num_classes`: Number of classes in the segmentation task.
                    Also controls the final output shape `(num_classes, height, width)`. Defaults to `21`
    - `backbone`: The neural network to use for extracting features. If `None`, then all params are set to
                `LRASPP_MobileNetV3` with `untrained` weights
    - `intermediate_layers`: Layers from `backbone` to be used for generating output maps. Assuming the backbone
        is of a `MobileNetV3`, default sets it to indices `[4, 16]` in `backbone.features`
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
    """
    if key is None:
        key = jr.PRNGKey(0)

    if num_classes is None:
        num_classes = 21

    if backbone is None:
        backbone = mobilenet_v3_large(dilated=True)
    if intermediate_layers is None:
        intermediate_layers = lambda x: [4, 16]

    backbone = backbone.features
    num_channels = [
        backbone.layers[y].out_channels for y in intermediate_layers(backbone)
    ]
    backbone = intermediate_layer_getter(backbone, intermediate_layers)
    model = LRASPP(
        backbone, num_channels[0], num_channels[1], num_classes=num_classes, key=key
    )
    if torch_weights:
        return load_torch_weights(model, torch_weights)

    return model

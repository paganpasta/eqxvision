from typing import Any, cast, Dict, List, Optional, Union

import equinox as eqx
import equinox.experimental as eqex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array

from ...layers import Activation
from ...utils import load_torch_weights, MODEL_URLS


_cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(eqx.Module):
    """A simple port of `torchvision.models.vgg`"""

    features: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    classifier: nn.Sequential

    def __init__(
        self,
        cfg: List[Union[str, int]] = None,
        num_classes: int = 1000,
        batch_norm: bool = True,
        dropout: float = 0.5,
        *,
        key: Optional["jax.random.PRNGKey"] = None
    ) -> None:
        """**Arguments:**

        - `cfg`: A list specifying the block configuration.
        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`.
        - `batch_norm` : If `True`, then `BatchNorm` is enabled in the architecture.
        - `dropout`: The probability parameter for `equinox.nn.Dropout`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super(VGG, self).__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 4)

        self.features = _make_layers(cfg, batch_norm, key=keys[0])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            [
                nn.Linear(512 * 7 * 7, 4096, key=keys[1]),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096, key=keys[2]),
                Activation(jnn.relu),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes, key=keys[3]),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        keys = jrandom.split(key, 2)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[1])
        return x


def _make_layers(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    key: "jax.random.PRNGKey" = None,
) -> nn.Sequential:

    layers: List[eqx.Module] = []
    in_channels = 3
    keys = jrandom.split(key=key, num=len(cfg) - cfg.count("M"))
    count = 0
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=1, key=keys[count]
            )
            if batch_norm:
                layers += [
                    conv2d,
                    eqex.BatchNorm(v, axis_name="batch"),
                    Activation(jnn.relu),
                ]
            else:
                layers += [conv2d, Activation(jnn.relu)]
            in_channels = v
            count += 1
    return nn.Sequential(layers)


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(cfg=_cfgs[cfg], batch_norm=batch_norm, **kwargs)
    return model


def vgg11(pretrained=False, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
        **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint.
    """
    model = _vgg("A", False, **kwargs)
    if pretrained:
        model = load_torch_weights(model, url=MODEL_URLS["vgg11"])
    return model


def vgg11_bn(**kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("A", True, **kwargs)


def vgg13(pretrained=False, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint.
    """
    model = _vgg("B", False, **kwargs)
    if pretrained:
        model = load_torch_weights(model, url=MODEL_URLS["vgg13"])
    return model


def vgg13_bn(**kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("B", True, **kwargs)


def vgg16(pretrained=False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint.
    """
    model = _vgg("D", False, **kwargs)
    if pretrained:
        model = load_torch_weights(model, url=MODEL_URLS["vgg16"])
    return model


def vgg16_bn(**kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("D", True, **kwargs)


def vgg19(pretrained=False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    **Arguments:**

    - `pretrained`: If `True`, the weights are loaded from `PyTorch` saved checkpoint.
    """
    model = _vgg("E", False, **kwargs)
    if pretrained:
        model = load_torch_weights(model, url=MODEL_URLS["vgg19"])
    return model


def vgg19_bn(**kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("E", True, **kwargs)

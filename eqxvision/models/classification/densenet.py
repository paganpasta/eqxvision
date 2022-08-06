from typing import Any, Optional, Sequence, Tuple, Union

import equinox as eqx
import equinox.experimental as eqxex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


class _DenseLayer(eqx.Module):
    norm1: eqxex.BatchNorm
    relu: nn.Lambda
    conv1: nn.Conv2d
    norm2: eqxex.BatchNorm
    conv2: nn.Conv2d
    dropout: nn.Dropout

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        key: "jax.random.PRNGKey",
    ) -> None:
        super().__init__()
        keys = jrandom.split(key, 2)
        self.norm1 = eqxex.BatchNorm(num_input_features, axis_name="batch")
        self.relu = nn.Lambda(jnn.relu)
        self.conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            use_bias=False,
            key=keys[0],
        )
        self.norm2 = eqxex.BatchNorm(bn_size * growth_rate, axis_name="batch")
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
        )
        self.dropout = nn.Dropout(p=float(drop_rate))

    def __call__(
        self, x: Union[Array, Sequence[Array]], *, key: "jax.random.PRNGKey"
    ) -> Array:
        if isinstance(x, Array):
            prev_features = [x]
        else:
            prev_features = x

        concated_features = jnp.concatenate(prev_features, axis=0)
        bottleneck_output = self.conv1(self.relu(self.norm1(concated_features)))
        new_features = self.conv2(self.relu(self.norm2(bottleneck_output)))
        new_features = self.dropout(new_features, key=key)
        return new_features


class _DenseBlock(eqx.Module):
    layers: Sequence[eqx.Module]
    num_layers: int

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()
        self.layers = []
        self.num_layers = num_layers
        keys = jrandom.split(key, num_layers)
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                key=keys[i],
            )
            self.layers.append(layer)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        features = [x]
        keys = jrandom.split(key, self.num_layers)
        for i in range(self.num_layers):
            new_features = self.layers[i](features, key=keys[i])
            features.append(new_features)
        return jnp.concatenate(features, 0)


class _Transition(eqx.Module):
    layers: nn.Sequential

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            [
                eqxex.BatchNorm(num_input_features, axis_name="batch"),
                nn.Lambda(jnn.relu),
                nn.Conv2d(
                    num_input_features,
                    num_output_features,
                    kernel_size=1,
                    stride=1,
                    use_bias=False,
                    key=key,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        return self.layers(x, key=key)


class DenseNet(eqx.Module):
    """A simple port of `torchvision.models.densenet`."""

    features: nn.Sequential
    classifier: nn.Linear

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> None:
        """
        **Arguments:**

        -   `growth_rate`: Number of filters to add in each layer (`k` in paper)
        -   `block_config`: Number of layers in each pooling block
        -   `num_init_features` - The number of filters to learn in the first convolution layer
        -   `bn_size`: Multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
        -   `drop_rate`: Dropout rate after each dense layer
        - `num_classes`: Number of classes in the classification task.
                            Also controls the final output shape `(num_classes,)`. Defaults to `1000`.
        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        # First convolution
        keys = jrandom.split(key, 2 * len(block_config) + 2)
        self.features = nn.Sequential(
            [
                nn.Conv2d(
                    3,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    use_bias=False,
                    key=keys[0],
                ),
                eqxex.BatchNorm(num_init_features, axis_name="batch"),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            keys = jrandom.split(keys[i * 2 + 1], 3)
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                key=keys[0],
            )
            self.features.layers.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    key=keys[i * 2 + 2],
                )
                self.features.layers.append(trans)
                num_features = num_features // 2

        # Final batch norm, relu and pooling
        self.features.layers.extend(
            [
                eqxex.BatchNorm(num_features, axis_name="batch"),
                nn.Lambda(jnn.relu),
                nn.AdaptiveAvgPool2d((1, 1)),
            ]
        )
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes, key=keys[-1])

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        out = self.features(x, key=key)
        out = jnp.ravel(out)
        out = self.classifier(out)
        return out


def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    **kwargs: Any,
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    """
    return _densenet(32, (6, 12, 24, 16), 64, **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    """
    return _densenet(48, (6, 12, 36, 24), 96, **kwargs)


def densenet169(**kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    """
    return _densenet(32, (6, 12, 32, 32), 64, **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    """
    return _densenet(32, (6, 12, 48, 32), 64, **kwargs)

from typing import Any, Callable, List, Optional, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


class GoogLeNet(eqx.Module):
    """A simple port of torchvision.models.GoogLeNet"""

    aux_logits: bool
    conv1: eqx.Module
    maxpool1: nn.MaxPool2d
    conv2: eqx.Module
    conv3: eqx.Module
    maxpool2: nn.MaxPool2d
    inception3a: eqx.Module
    inception3b: eqx.Module
    maxpool3: nn.MaxPool2d
    inception4a: eqx.Module
    inception4b: eqx.Module
    inception4c: eqx.Module
    inception4d: eqx.Module
    inception4e: eqx.Module
    maxpool4: nn.MaxPool2d
    inception5a: eqx.Module
    inception5b: eqx.Module
    aux1: eqx.Module
    aux2: eqx.Module
    avgpool: nn.AdaptiveAvgPool2d
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        blocks: Optional[List[Callable[..., eqx.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        *,
        key: Optional["jax.random.PRNGKey"] = None
    ) -> None:
        """
        **Arguments:**

        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`.
        - `aux_logits`: If `True`, two auxiliary branches are added to the network. Defaults to `True`.
        - `blocks`: Blocks for constructing the network.
        - `dropout`: Dropout applied on the `main` branch. Defaults to `0.2`.
        - `dropout_aux`: Dropout applied on the `aux` branches. Defaults to `0.7`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        """
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 20)

        self.aux_logits = aux_logits
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3, key=keys[0])
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = conv_block(64, 64, kernel_size=1, key=keys[1])
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1, key=keys[2])
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32, key=keys[3])
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64, key=keys[4])
        self.maxpool3 = nn.MaxPool2d(3, stride=2)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64, key=keys[5])
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64, key=keys[6])
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64, key=keys[7])
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64, key=keys[8])
        self.inception4e = inception_block(
            528, 256, 160, 320, 32, 128, 128, key=keys[9]
        )
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.inception5a = inception_block(
            832, 256, 160, 320, 32, 128, 128, key=keys[10]
        )
        self.inception5b = inception_block(
            832, 384, 192, 384, 48, 128, 128, key=keys[11]
        )

        self.aux1 = None
        self.aux2 = None
        if aux_logits:
            self.aux1 = inception_aux_block(
                512, num_classes, dropout=dropout_aux, key=keys[12]
            )
            self.aux2 = inception_aux_block(
                528, num_classes, dropout=dropout_aux, key=keys[13]
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes, key=keys[14])

    def __call__(
        self, x: Array, *, key: "jax.random.PRNGKey"
    ) -> Union[Array, Optional[Array], Optional[Array]]:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        if key is None:
            raise RuntimeError("The model requires a PRNGKey.")
        keys = jrandom.split(key, 14)
        # N x 3 x 224 x 224
        x = self.conv1(x, key=keys[0])
        # N x 64 x 112 x 112
        x = self.maxpool1(x, key=keys[1])
        # N x 64 x 56 x 56
        x = self.conv2(x, key=keys[2])
        # N x 64 x 56 x 56
        x = self.conv3(x, key=keys[3])
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x, key=keys[4])
        # N x 256 x 28 x 28
        x = self.inception3b(x, key=keys[5])
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x, key=keys[6])
        # N x 512 x 14 x 14
        if self.aux_logits:
            aux1 = self.aux1(x, key=keys[7])

        x = self.inception4b(x, key=keys[8])
        # N x 512 x 14 x 14
        x = self.inception4c(x, key=keys[9])
        # N x 512 x 14 x 14
        x = self.inception4d(x, key=keys[10])
        # N x 528 x 14 x 14
        if self.aux_logits:
            aux2 = self.aux2(x, key=keys[11])  # Key here, a bad thing?

        x = self.inception4e(x, key=keys[12])
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x, key=keys[13])
        # N x 832 x 7 x 7
        x = self.inception5b(x, key=keys[14])
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = jnp.ravel(x)
        # N x 1024
        x = self.dropout(x, key=keys[15])
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.aux_logits:
            return x, aux2, aux1
        else:
            return x


class Inception(eqx.Module):
    branch1: eqx.Module
    branch2: nn.Sequential
    branch3: nn.Sequential
    branch4: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., eqx.Module]] = None,
        *,
        key: jax.random.PRNGKey = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        keys = jrandom.split(key, 5)
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, key=keys[0])
        self.branch2 = nn.Sequential(
            [
                conv_block(in_channels, ch3x3red, kernel_size=1, key=keys[1]),
                conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1, key=keys[2]),
            ]
        )

        self.branch3 = nn.Sequential(
            [
                conv_block(in_channels, ch5x5red, kernel_size=1, key=keys[3]),
                # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
                # Please see https://github.com/pytorch/vision/issues/906 for details.
                conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1, key=keys[4]),
            ]
        )

        self.branch4 = nn.Sequential(
            [
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, key=keys[5]),
            ]
        )

    def __call__(self, x: Array, *, key: jax.random.PRNGKey = None) -> Array:
        keys = jrandom.split(key, 4)
        branch1 = self.branch1(x, key=keys[0])
        branch2 = self.branch2(x, key=keys[1])
        branch3 = self.branch3(x, key=keys[2])
        branch4 = self.branch4(x, key=keys[3])

        outputs = jnp.concatenate([branch1, branch2, branch3, branch4], axis=0)
        return outputs


class InceptionAux(eqx.Module):
    conv: eqx.Module
    fc1: nn.Linear
    fc2: nn.Linear
    dropout: nn.Dropout
    avgpool: nn.AdaptiveAvgPool2d

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., eqx.Module]] = None,
        dropout: float = 0.7,
        *,
        key: jax.random.PRNGKey = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        keys = jrandom.split(key, 3)
        self.conv = conv_block(in_channels, 128, kernel_size=1, key=keys[0])
        self.fc1 = nn.Linear(2048, 1024, key=keys[1])
        self.fc2 = nn.Linear(1024, num_classes, key=keys[2])
        self.dropout = nn.Dropout(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (4, 4),
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey" = None) -> Array:
        keys = jrandom.split(key, 2)
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.avgpool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x, key=keys[0])
        # N x 128 x 4 x 4
        x = jnp.ravel(x)
        # N x 2048
        x = jnn.relu(self.fc1(x))
        # N x 1024
        x = self.dropout(x, key=keys[1])
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(eqx.Module):
    conv: nn.Conv2d
    bn: eqx.experimental.BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: jax.random.PRNGKey = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, use_bias=False, key=key, **kwargs
        )
        self.bn = eqx.experimental.BatchNorm(out_channels, axis_name="batch", eps=0.001)

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        x = self.conv(x)
        x = self.bn(x, key=key)
        return jnn.relu(x)


def googlenet(**kwargs: Any) -> GoogLeNet:
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    The required minimum input size of the model is 15x15.
    """
    return GoogLeNet(**kwargs)

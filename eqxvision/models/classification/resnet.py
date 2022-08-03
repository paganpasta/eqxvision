from typing import Any, Callable, List, Optional, Sequence, Type, Union

import equinox as eqx
import equinox.experimental as eqex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, key=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def conv1x1(in_planes, out_planes, stride=1, key=None):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class ResNetBasicBlock(eqx.Module):
    expansion: int
    conv1: eqx.Module
    bn1: eqx.Module
    relu: Callable
    conv2: eqx.Module
    bn2: eqx.Module
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        super(ResNetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = eqex.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jrandom.split(key, 2)
        self.expansion = 1
        self.conv1 = conv3x3(inplanes, planes, stride, key=keys[0])
        self.bn1 = norm_layer(planes, axis_name="batch")
        self.relu = jnn.relu
        self.conv2 = conv3x3(planes, planes, key=keys[1])
        self.bn2 = norm_layer(planes, axis_name="batch")
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNetBottleneck(eqx.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion: int
    conv1: eqx.Module
    bn1: eqx.Module
    conv2: eqx.Module
    bn2: eqx.Module
    conv3: eqx.Module
    bn3: eqx.Module
    relu: Callable
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        super(ResNetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = eqex.BatchNorm
        self.expansion = 4
        keys = jrandom.split(key, 3)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, key=keys[0])
        self.bn1 = norm_layer(width, axis_name="batch")
        self.conv2 = conv3x3(width, width, stride, groups, dilation, key=keys[1])
        self.bn2 = norm_layer(width, axis_name="batch")
        self.conv3 = conv1x1(width, planes * self.expansion, key=keys[2])
        self.bn3 = norm_layer(planes * self.expansion, axis_name="batch")
        self.relu = jnn.relu
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def __call__(self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


EXPANSIONS = {ResNetBasicBlock: 1, ResNetBottleneck: 4}


class ResNet(eqx.Module):
    """A simple port of `torchvision.models.resnet`"""

    inplanes: int
    dilation: int
    groups: Sequence[int]
    base_width: int
    conv1: eqx.Module
    bn1: eqx.Module
    relu: jnn.relu
    maxpool: eqx.Module
    layer1: eqx.Module
    layer2: eqx.Module
    layer3: eqx.Module
    layer4: eqx.Module
    avgpool: eqx.Module
    fc: eqx.Module

    def __init__(
        self,
        block: Type[Union[ResNetBasicBlock, ResNetBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual=False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: List[bool] = None,
        norm_layer: Any = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        """**Arguments:**

        - `block`: `Bottleneck` or `BasicBlock` for constructing the network.
        - `layers`: A list containing number of `blocks` at different levels.
        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`.
        - `zero_init_residual`: **Ignored**.
        - `groups`: Number of groups to form along the feature depth. Defaults to `1`.
        - `width_per_group`: Increases width of `block` by a factor of `width_per_group/64`.
        Defaults to `64`.
        - `replace_stride_with_dilation`: Replacing `2x2` strides with dilated convolution. Defaults to None.
        - `norm_layer`: Normalisation to be applied on the inputs. Defaults to `equinox.BatchNorm`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

        ??? Failure "Exceptions:"

            - `NotImplementedError`: If a `norm_layer` other than `equinox.experimental.BatchNorm` is used.
            - `ValueError`: If `replace_stride_with_convolution` is not `None` or a `3-tuple`.

        """
        super(ResNet, self).__init__()
        if not norm_layer:
            norm_layer = eqex.BatchNorm

        if eqex.BatchNorm != norm_layer:
            raise NotImplementedError(
                f"{type(norm_layer)} is not currently supported. Use `eqx.experimental.BatchNorm` instead."
            )
        if key is None:
            key = jrandom.PRNGKey(0)

        keys = jrandom.split(key, 6)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )
        self.bn1 = norm_layer(input_size=self.inplanes, axis_name="batch")
        self.relu = jnn.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, key=keys[1])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
            key=keys[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
            key=keys[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
            key=keys[4],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * EXPANSIONS[block], num_classes, key=keys[5])
        # TODO: Zero initialize BNs as per torchvision

    def _make_layer(
        self, block, planes, blocks, norm_layer, stride=1, dilate=False, key=None
    ):
        keys = jrandom.split(key, blocks + 1)
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * EXPANSIONS[block]:
            downsample = nn.Sequential(
                [
                    conv1x1(
                        self.inplanes, planes * EXPANSIONS[block], stride, key=keys[0]
                    ),
                    norm_layer(planes * EXPANSIONS[block], axis_name="batch"),
                ]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                key=keys[1],
            )
        )
        self.inplanes = planes * EXPANSIONS[block]
        for block_idx in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    key=keys[block_idx + 1],
                )
            )

        return nn.Sequential(layers)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels.
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`.
        """
        if key is None:
            raise RuntimeError("The model requires a PRNGKey.")
        keys = jrandom.split(key, 6)
        x = self.conv1(x, key=keys[0])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, key=keys[1])
        x = self.layer2(x, key=keys[2])
        x = self.layer3(x, key=keys[3])
        x = self.layer4(x, key=keys[4])

        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x, key=keys[5])

        return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet(ResNetBasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet(ResNetBasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet(ResNetBottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet(ResNetBottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(ResNetBottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(ResNetBottleneck, [3, 4, 23, 3], **kwargs)

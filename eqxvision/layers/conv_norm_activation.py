from typing import Callable, Optional

import equinox.experimental as eqxex
import equinox.nn as nn
import jax
import jax.nn as jnn


class ConvNormActivation(nn.Sequential):
    """A simple port of `torchvision.ops.misc.ConvNormActivation`.

    Packs `convolution` -> `normalisation` -> `activation` into one easy to use module.
    """

    out_channels: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable] = eqxex.BatchNorm,
        activation_layer: Optional[Callable] = jnn.relu,
        dilation: int = 1,
        bias: Optional[bool] = None,
        *,
        key: "jax.random.PRNGKey" = None
    ) -> None:
        """

        - `in_channels`: Number of channels in the input image
        - `out_channels`: Number of channels produced by the Convolution-Normalzation-Activation block
        - `kernel_size`: Size of the convolution kernel. Defaults to `3`
        - `stride`: Stride of the convolution. Defaults to `1`
        - `padding`: Padding added to all four sides of the input. Defaults to `None`,
            in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        - `groups`: Number of blocked connections from input channels to output channels. Defaults to `1`
        - `norm_layer`: Norm layer that will be stacked on top of the convolution layer. If ``None``
            this layer wont be used. Defaults to ``eqx.experimental.BatchNorm``
        - `activation_layer`: Activation function which will be stacked on top of the normalization layer
            (if not None), otherwise on top of the conv layer
            If ``None`` this layer wont be used. Defaults to ``jax.nn.relu``
        - `dilation`: Spacing between kernel elements. Defaults to `1`
        - `bias`: If `True`, bias is used in the convolution layer. By default, biases are included #
            if ``norm_layer is None``
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                use_bias=bias,
                key=key,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels, axis_name="batch"))
        if activation_layer is not None:
            layers.append(nn.Lambda(activation_layer))
        super().__init__(layers)
        self.out_channels = out_channels

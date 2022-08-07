from typing import Any, Callable, List, Optional

import equinox as eqx
import equinox.experimental as eqxex
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


def channel_shuffle(x: Array, groups: int) -> Array:
    num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = jnp.reshape(x, (groups, channels_per_group, height, width))
    x = jnp.transpose(x, axes=(1, 0, 2, 3))
    x = jnp.reshape(x, (-1, height, width))
    return x


class InvertedResidual(eqx.Module):
    stride: int
    branch1: nn.Sequential
    branch2: nn.Sequential

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        *,
        key: "jax.random.PRNGKey" = None,
    ) -> None:
        super().__init__()

        keys = jrandom.split(key, 5)

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")

        branch_features = oup // 2
        assert (stride != 1) or (inp == branch_features << 1)

        self.stride = stride
        if stride > 1:
            self.branch1 = nn.Sequential(
                [
                    self.depthwise_conv(
                        inp,
                        inp,
                        kernel_size=3,
                        stride=self.stride,
                        padding=1,
                        key=keys[0],
                    ),
                    eqxex.BatchNorm(inp, axis_name="batch"),
                    nn.Conv2d(
                        inp,
                        branch_features,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        use_bias=False,
                        key=keys[1],
                    ),
                    eqxex.BatchNorm(branch_features, axis_name="batch"),
                    nn.Lambda(jnn.relu),
                ]
            )
        else:
            self.branch1 = nn.Sequential([nn.Identity])

        self.branch2 = nn.Sequential(
            [
                nn.Conv2d(
                    inp if (self.stride > 1) else branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=False,
                    key=keys[2],
                ),
                eqxex.BatchNorm(branch_features, axis_name="batch"),
                nn.Lambda(jnn.relu),
                self.depthwise_conv(
                    branch_features,
                    branch_features,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    key=keys[3],
                ),
                eqxex.BatchNorm(branch_features, axis_name="batch"),
                nn.Conv2d(
                    branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=False,
                    key=keys[4],
                ),
                eqxex.BatchNorm(branch_features, axis_name="batch"),
                nn.Lambda(jnn.relu),
            ]
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        key=None,
    ) -> nn.Conv2d:
        return nn.Conv2d(
            i, o, kernel_size, stride, padding, use_bias=bias, groups=i, key=key
        )

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        if self.stride == 1:
            x1, x2 = jnp.split(x, 2, axis=0)
            out = jnp.concatenate((x1, self.branch2(x2)), axis=0)
        else:
            out = jnp.concatenate((self.branch1(x), self.branch2(x)), axis=0)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(eqx.Module):
    """A simple port of `torchvision.models.shufflenetv2`"""

    conv1: nn.Sequential
    maxpool: nn.MaxPool2d
    stage2: nn.Sequential
    stage3: nn.Sequential
    stage4: nn.Sequential
    conv5: nn.Sequential
    pool: nn.AdaptiveAvgPool2d
    fc: nn.Linear

    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., eqx.Module] = InvertedResidual,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> None:
        """**Arguments:**

        - stages_repeats: Number of times a block is repeated for each stage
        - stages_out_channels: Output at each stage
        - num_classes: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - inverted_residual: Network structure
        - key: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 2)

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")

        input_channels = 3
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            [
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    3,
                    2,
                    1,
                    use_bias=False,
                    key=keys[0],
                ),
                eqxex.BatchNorm(output_channels, axis_name="batch"),
                nn.Lambda(jnn.relu),
            ]
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, stages_out_channels[1:]
        ):
            keys = jrandom.split(keys[1], 2)
            seq = [inverted_residual(input_channels, output_channels, 2, key=keys[0])]
            for i in range(repeats - 1):
                keys = jrandom.split(keys[1], 2)
                seq.append(
                    inverted_residual(output_channels, output_channels, 1, key=keys[0])
                )
            setattr(self, name, nn.Sequential(seq))
            input_channels = output_channels

        keys = jrandom.split(keys[1], 2)
        output_channels = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            [
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    1,
                    1,
                    0,
                    use_bias=False,
                    key=keys[0],
                ),
                eqxex.BatchNorm(output_channels, axis_name="batch"),
                nn.Lambda(jnn.relu),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channels, num_classes, key=keys[1])

    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 5)
        x = self.conv1(x, key=keys[0])
        x = self.maxpool(x)
        x = self.stage2(x, key=keys[1])
        x = self.stage3(x, key=keys[2])
        x = self.stage4(x, key=keys[3])
        x = self.conv5(x, key=keys[4])
        x = jnp.ravel(self.pool(x))
        x = self.fc(x)
        return x


def _shufflenetv2(*args: Any, **kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)
    return model


def shufflenet_v2_x0_5(**kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return _shufflenetv2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(**kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return _shufflenetv2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(**kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return _shufflenetv2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(**kwargs: Any) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    """
    return _shufflenetv2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)

import warnings
from functools import partial
from typing import Any, Callable, List, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.custom_types import Array

from ...layers import DropPath, LayerNorm2d, Linear2d, MlpProjection
from ...utils import load_torch_weights


def _func_dropout(x, p, key):
    q = 1 - p
    mask = jr.bernoulli(key, q, x.shape)
    return jnp.where(mask, x / q, 0)


def _patch_merging_pad(x: Array) -> Array:
    _, H, W = x.shape
    x = jnp.pad(x, ((0, 0), (0, H % 2), (0, W % 2)))
    x0 = x[:, 0::2, 0::2]  # ... H/2 W/2 C
    x1 = x[:, 1::2, 0::2]  # ... H/2 W/2 C
    x2 = x[:, 0::2, 1::2]  # ... H/2 W/2 C
    x3 = x[:, 1::2, 1::2]  # ... H/2 W/2 C
    x = jnp.concatenate([x0, x1, x2, x3], axis=0)  # 4*C H/2 W/2
    return x


def _get_relative_position_bias(
    relative_position_bias_table: Array,
    relative_position_index: Array,
    window_size: List[int],
) -> Array:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = jnp.reshape(relative_position_bias, (N, N, -1))
    relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
    return relative_position_bias


class _PatchMerging(eqx.Module):
    reduction: Linear2d
    norm: Callable

    def __init__(
        self,
        dim: int,
        norm_layer: Callable = LayerNorm2d,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = Linear2d(4 * dim, 2 * dim, use_bias=False, key=key)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


class _PatchMergingV2(eqx.Module):
    reduction: Linear2d
    norm: Callable

    def __init__(
        self,
        dim: int,
        norm_layer: Callable = LayerNorm2d,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        self.norm = norm_layer(2 * dim)
        self.reduction = Linear2d(4 * dim, 2 * dim, use_bias=False, key=key)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def _shifted_window_attention(
    x: Array,
    qkv_weight: Array,
    proj_weight: Array,
    relative_position_bias: Array,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Array] = None,
    proj_bias: Optional[Array] = None,
    logit_scale: Optional["jax.numpy.ndarray"] = None,
    key: "jax.random.PRNGKey" = None,
):
    """Like `torchvision.models.swin_transformer.shifted_window_attention`."""
    x = jnp.transpose(x, (1, 2, 0))
    H, W, C = x.shape
    # TODO: Support flexible padding
    # pad feature maps to multiples of window size
    # pad_r = jnp.mod((window_size[1] - jnp.mod(W, window_size[1])), window_size[1])
    # pad_b = jnp.mod((window_size[0] - jnp.mod(H, window_size[0])), window_size[0])
    # x = jnp.pad(x, ((0, pad_r), (0, pad_b), (0, 0)))
    pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(0, 1))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = jnp.reshape(
        x,
        (
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
            C,
        ),
    )
    x = jnp.reshape(
        jnp.transpose(x, (0, 2, 1, 3, 4)),
        (num_windows, window_size[0] * window_size[1], C),
    )  # nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.copy()
        length = qkv_bias.size // 3
        set_zero_slice = lambda c, a: (c.at[a].set(0), a)
        qkv_bias, _ = jax.lax.scan(
            set_zero_slice, qkv_bias, jnp.arange(length, 2 * length)
        )  # qkv_bias[length:2*length].zero_()
    qkv = (
        jnp.matmul(x, jnp.transpose(qkv_weight)) + qkv_bias
    )  # F.linear(x, qkv_weight, qkv_bias)
    qkv = jnp.transpose(
        jnp.reshape(qkv, (x.shape[0], x.shape[1], 3, num_heads, C // num_heads)),
        (2, 0, 3, 1, 4),
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = (q / jnp.linalg.norm(q, ord=2, axis=0)) @ jnp.transpose(
            (k / jnp.linalg.norm(k, ord=2, axis=0)), (0, 1, 3, 2)
        )
        # attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = jnp.exp(
            jax.lax.clamp(min=-jnp.inf, x=logit_scale, max=jnp.log(100.0))
        )
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q @ jnp.transpose(k, (0, 1, 3, 2))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_00 = jnp.ones((pad_H - window_size[0], pad_W - window_size[1])) * 0
        attn_01 = jnp.ones((pad_H - window_size[0], window_size[1] - shift_size[1])) * 1
        attn_02 = jnp.ones((pad_H - window_size[0], shift_size[1])) * 2
        attn_10 = jnp.ones((window_size[0] - shift_size[0], pad_W - window_size[1])) * 3
        attn_11 = (
            jnp.ones((window_size[0] - shift_size[0], window_size[1] - shift_size[1]))
            * 4
        )
        attn_12 = jnp.ones((window_size[0] - shift_size[0], shift_size[1])) * 5
        attn_20 = jnp.ones((shift_size[0], pad_W - window_size[1])) * 6
        attn_21 = jnp.ones((shift_size[0], window_size[1] - shift_size[1])) * 7
        attn_22 = jnp.ones((shift_size[0], shift_size[1])) * 8

        attn_mask = jnp.concatenate(
            jnp.concatenate(
                (
                    jnp.concatenate((attn_00, attn_01, attn_02), axis=1),
                    jnp.concatenate((attn_10, attn_11, attn_12), axis=1),
                    jnp.concatenate((attn_20, attn_21, attn_22), axis=1),
                )
            ),
            axis=0,
        )

        attn_mask = jnp.reshape(
            attn_mask,
            (
                pad_H // window_size[0],
                window_size[0],
                pad_W // window_size[1],
                window_size[1],
            ),
        )
        attn_mask = jnp.reshape(
            jnp.transpose(attn_mask, (0, 2, 1, 3)),
            (num_windows, window_size[0] * window_size[1]),
        )
        attn_mask = jnp.expand_dims(attn_mask, axis=1) - jnp.expand_dims(
            attn_mask, axis=2
        )
        attn_mask = jnp.where(attn_mask == 0, 0.0, -100.0)
        attn = jnp.reshape(
            attn,
            (x.shape[0] // num_windows, num_windows, num_heads, x.shape[1], x.shape[1]),
        )
        attn = attn + jnp.expand_dims(attn_mask, axis=1)
        attn = jnp.reshape(attn, (-1, num_heads, x.shape[1], x.shape[1]))

    attn = jnn.softmax(attn, axis=-1)

    attn = _func_dropout(attn, p=attention_dropout, key=key)

    x = jnp.reshape(
        jnp.transpose((attn @ v), (0, 2, 1, 3)), (x.shape[0], x.shape[1], C)
    )
    x = x @ jnp.transpose(proj_weight, (1, 0)) + proj_bias
    x = _func_dropout(x, p=dropout, key=key)

    # reverse windows
    x = jnp.reshape(
        x,
        (
            pad_H // window_size[0],
            pad_W // window_size[1],
            window_size[0],
            window_size[1],
            C,
        ),
    )
    x = jnp.reshape(jnp.transpose(x, (0, 2, 1, 3, 4)), (pad_H, pad_W, C))

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(shift_size[0], shift_size[1]), axis=(0, 1))

    x = jnp.reshape(
        jnp.take(jnp.reshape(x, (-1, C)), indices=jnp.arange(H * W), axis=0), (H, W, C)
    )
    return jnp.transpose(x, (2, 0, 1))


class _ShiftedWindowAttention(eqx.Module):

    window_size: List[int]
    shift_size: List[int]
    num_heads: int
    attention_dropout: float
    dropout: float
    relative_position_bias_table: jnp.ndarray
    relative_position_index: jnp.ndarray
    qkv: nn.Linear
    proj: nn.Linear

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")

        keys = jr.split(key, 3)

        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = Linear2d(dim, dim * 3, use_bias=qkv_bias, key=keys[0])
        self.proj = Linear2d(dim, dim, use_bias=proj_bias, key=keys[1])

        self.relative_position_bias_table = self.define_relative_position_bias_table(
            key=keys[2]
        )
        self.relative_position_index = self.define_relative_position_index()

    def define_relative_position_bias_table(self, key):
        return jr.truncated_normal(
            key=key,
            shape=(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
            lower=2,
            upper=2,
        )

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = jax.vmap(jnp.ravel)(coords)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = jnp.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        jnp.stack(
            (
                (relative_coords[:, :, 0] + self.window_size[0] - 1)
                * 2
                * self.window_size[1]
                - 1,
                relative_coords[:, :, 1] + self.window_size[1] - 1,
            ),
            axis=-1,
        )
        relative_position_index = jnp.ravel(relative_coords.sum(axis=-1))  # Wh*Ww*Wh*Ww
        return relative_position_index

    def get_relative_position_bias(self) -> Array:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input array of shape `(H, W, C)`
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`

        **Returns:**

        An array of shape `(H, W, C)`
        """
        relative_position_bias = self.get_relative_position_bias()
        return _shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            key=key,
        )


class _ShiftedWindowAttentionV2(eqx.Module):

    window_size: List[int]
    shift_size: List[int]
    num_heads: int
    attention_dropout: float
    dropout: float
    logit_scale: jnp.ndarray
    relative_position_bias_table: jnp.ndarray
    relative_position_index: jnp.ndarray
    qkv: nn.Linear
    proj: nn.Linear
    cpb_mlp: nn.Sequential

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")

        keys = jr.split(key, 3)

        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = Linear2d(dim, dim * 3, use_bias=qkv_bias, key=keys[0])
        self.proj = Linear2d(dim, dim, use_bias=proj_bias, key=keys[1])

        self.relative_position_bias_table = self.define_relative_position_bias_table(
            key=keys[2]
        )
        self.relative_position_index = self.define_relative_position_index()

        self.logit_scale = jnp.log(10 * jnp.ones((num_heads, 1, 1)))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            [
                nn.Lambda(partial(jnp.transpose, axes=(2, 0, 1))),
                Linear2d(2, 512, use_bias=True, key=keys[1]),
                nn.Lambda(jnn.relu),
                Linear2d(512, num_heads, use_bias=False, key=keys[2]),
                nn.Lambda(partial(jnp.transpose, axes=(0, 1, 2))),
            ]
        )
        if qkv_bias:
            length = self.qkv.bias.size // 3
            set_zero_slice = lambda c, a: (c.at[a].set(0), a)
            qkv_bias, _ = jax.lax.scan(
                set_zero_slice, self.qkv.bias, jnp.arange(length, 2 * length)
            )
            self.qkv = eqx.tree_at(lambda l: l.bias, self.qkv, qkv_bias)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = jax.vmap(jnp.ravel)(coords)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = jnp.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        jnp.stack(
            (
                (relative_coords[:, :, 0] + self.window_size[0] - 1)
                * 2
                * self.window_size[1]
                - 1,
                relative_coords[:, :, 1] + self.window_size[1] - 1,
            ),
            axis=-1,
        )
        relative_position_index = jnp.ravel(relative_coords.sum(axis=-1))  # Wh*Ww*Wh*Ww
        return relative_position_index

    def define_relative_position_bias_table(self, key):
        # get relative_coords_table
        relative_coords_h = jnp.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=jnp.float32
        )
        relative_coords_w = jnp.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=jnp.float32
        )
        relative_coords_table = jnp.stack(
            jnp.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        relative_coords_table = jnp.transpose(relative_coords_table, (1, 2, 0))

        relative_coords_table = jnp.stack(
            [
                relative_coords_table[:, :, 0] / self.window_size[0] - 1,
                relative_coords_table[:, :, 1] / self.window_size[1] - 1,
            ],
            axis=-1,
        )
        relative_coords_table = 8 * relative_coords_table  # normalize to -8, 8
        relative_coords_table = (
            jnp.sign(relative_coords_table)
            * jnp.log2(jnp.abs(relative_coords_table) + 1.0)
            / 3.0
        )
        return relative_coords_table

    def get_relative_position_bias(self) -> Array:
        relative_position_bias = _get_relative_position_bias(
            jnp.reshape(
                self.cpb_mlp(self.relative_position_bias_table), (-1, self.num_heads)
            ),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * jnn.sigmoid(relative_position_bias)
        return relative_position_bias

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input array of shape `(H, W, C)`
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`

        **Returns:**

        An array of shape `(H, W, C)`
        """
        relative_position_bias = self.get_relative_position_bias()
        return _shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            key=key,
        )


class _SwinTransformerBlock(eqx.Module):
    norm1: Callable
    attn: eqx.Module
    stochastic_depth: DropPath
    norm2: Callable
    mlp: MlpProjection

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., eqx.Module] = LayerNorm2d,
        attn_layer: Callable[..., eqx.Module] = _ShiftedWindowAttention,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        keys = jr.split(key, 2)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            key=keys[0],
        )
        self.stochastic_depth = DropPath(stochastic_depth_prob, mode="local")
        self.norm2 = norm_layer(dim)
        self.mlp = MlpProjection(
            dim,
            int(dim * mlp_ratio),
            dim,
            lin_layer=Linear2d,
            act_layer=jnn.gelu,
            drop=dropout,
            key=keys[1],
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        keys = jr.split(key, 4)
        x = x + self.stochastic_depth(
            self.attn(self.norm1(x), key=keys[0]), key=keys[1]
        )
        x = x + self.stochastic_depth(self.mlp(self.norm2(x), key=keys[2]), key=keys[3])
        return x


class _SwinTransformerBlockV2(eqx.Module):

    norm1: Callable
    attn: eqx.Module
    stochastic_depth: DropPath
    norm2: Callable
    mlp: MlpProjection

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., eqx.Module] = LayerNorm2d,
        attn_layer: Callable[..., eqx.Module] = _ShiftedWindowAttentionV2,
        *,
        key: "jax.random.PRNGKey" = None,
    ):
        super().__init__()
        keys = jr.split(key, 2)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            key=keys[0],
        )
        self.stochastic_depth = DropPath(stochastic_depth_prob, mode="local")
        self.norm2 = norm_layer(dim)
        self.mlp = MlpProjection(
            dim,
            int(dim * mlp_ratio),
            dim,
            lin_layer=Linear2d,
            act_layer=jnn.gelu,
            drop=dropout,
            key=keys[1],
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        keys = jr.split(key, 4)
        x = x + self.stochastic_depth(
            self.norm1(self.attn(x, key=keys[0])), key=keys[1]
        )
        x = x + self.stochastic_depth(self.norm2(self.mlp(x, key=keys[2])), key=keys[3])
        return x


class SwinTransformer(eqx.Module):
    """A simple port of `torchvision.models.swin_transformer`."""

    features: nn.Sequential
    norm: Callable
    avgpool: nn.AdaptiveAvgPool2d
    head: nn.Linear

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Callable = None,
        block: "eqx.Module" = None,
        downsample_layer: "eqx.Module" = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        """**Arguments:**

        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(4, 4)`
        - `embed_dim`: The dimension of the resulting embedding of the patch
        - `depths`: Depth of each Swin Transformer layer
        - `num_heads`: Number of attention heads in different layers
        - `window_size`: Window size
        - `mlp_ratio`:  Ratio of mlp hidden dim to embedding dim. Defaults to `4.0`
        - `dropout`: Dropout rate. Defaults to `0.0`
        - `attention_dropout`: Attention dropout rate. Defaults to `0.0`
        - `stochastic_depth_prob`:  Stochastic depth rate. Defaults to `0.1`
        - `num_classes`:  Number of classes in the classification task.
                         Also controls the final output shape `(num_classes,)`
        - `norm_layer`: Normalisation applied to the intermediate outputs. Defaults to `LayerNorm2d`
        - `block`: The SwinTransformer-v1/v2 block to use. Defaults to `_SwinTransformerBlock` which is used in `v1`
        - `downsample_layer`: Downsample layer (patch merging). Defaults to `_PatchMerging` which is used in `v1`
        - `key`:  A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

        !!! warning "Note"

            Currently, input image dimensions are required to be divisible by the `window_size` for
            each level of `depth`. For example, input image shape `(3, 224, 224)` works for window_size `(7, 7)`
            and `(3, 256, 256)` works for `(8, 8)`.
        """
        super().__init__()
        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, 2)

        if block is None:
            block = _SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-5)
        if downsample_layer is None:
            downsample_layer = _PatchMerging

        layers: List[eqx.Module] = []
        layers.append(
            nn.Sequential(
                [
                    nn.Conv2d(
                        3,
                        embed_dim,
                        kernel_size=(patch_size[0], patch_size[1]),
                        stride=(patch_size[0], patch_size[1]),
                        key=keys[0],
                    ),
                    norm_layer(embed_dim),
                ]
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[eqx.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                keys = jr.split(keys[1], 2)
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        key=keys[0],
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                keys = jr.split(keys[1], 2)
                layers.append(downsample_layer(dim, norm_layer, key=keys[0]))
        self.features = nn.Sequential(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes, key=keys[1])

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jr.split(key, 2)
        x = self.features(x, key=keys[0])
        x = self.norm(x)
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.head(x, key=keys[1])
        return x


def _swin_transformer(
    arch: str,
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    torch_weights: str,
    **kwargs: Any,
) -> SwinTransformer:

    warnings.warn(
        "Currently, dynamic padding of the input is not supported! "
        + "Please make sure that the input is a multiple of window_size."
    )

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )
    if torch_weights:
        model = load_torch_weights(model, torch_weights=torch_weights)
    return model


def swin_t(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    return _swin_transformer(
        arch="swin_t",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        torch_weights=torch_weights,
        **kwargs,
    )


def swin_s(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_small architecture from
    [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    return _swin_transformer(
        arch="swin_s",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        torch_weights=torch_weights,
        **kwargs,
    )


def swin_b(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_base architecture from
    [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    return _swin_transformer(
        arch="swin_b",
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        torch_weights=torch_weights,
        **kwargs,
    )


def swin_v2_t(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny architecture from
    [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    return _swin_transformer(
        arch="swin_v2_t",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=_SwinTransformerBlockV2,
        downsample_layer=_PatchMergingV2,
        torch_weights=torch_weights,
        **kwargs,
    )


def swin_v2_s(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_small architecture from
    [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    return _swin_transformer(
        arch="swin_v2_s",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=_SwinTransformerBlockV2,
        torch_weights=torch_weights,
        downsample_layer=_PatchMergingV2,
        **kwargs,
    )


def swin_v2_b(torch_weights: str = None, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_base architecture from
    [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883).

    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """

    return _swin_transformer(
        arch="swin_v2_b",
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=_SwinTransformerBlockV2,
        downsample_layer=_PatchMergingV2,
        torch_weights=torch_weights,
        **kwargs,
    )

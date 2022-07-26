from typing import Optional, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox.custom_types import Array


class PatchEmbed(eqx.Module):
    """2D Image to Patch Embedding ported from Timm"""

    img_size: Tuple[int]
    patch_size: Tuple[int]
    grid_size: Tuple[int]
    num_patches: int
    flatten: bool
    proj: eqx.nn.Conv2d
    norm: eqx.Module

    def __init__(
        self,
        img_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: eqx.Module = None,
        flatten: bool = True,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`.
        - `in_chans`: Number of input channels. Defaults to `3`.
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`.
        - `norm_layer`: The normalisation to be applied on an input. Defaults to None.
        - `flatten`: If enabled, the `2d` patches are flattened to `1d`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if key is None:
            key = jrandom.PRNGKey(0)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, key=key
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape`(in_chans, img_size[0], img_size[1])`.
        - `key`: Ignored.
        """
        C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image height ({H},{W}) doesn't match model ({self.img_size})."
            )

        x = self.proj(x)
        if self.flatten:
            x = jax.vmap(jnp.ravel)(x)
            x = jnp.moveaxis(x, 0, -1)  # CHW -> NC
        x = self.norm(x)
        return x

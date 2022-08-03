import logging
import os
import warnings
from pathlib import Path
from typing import NewType

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu


try:
    import torch
except ImportError:
    warnings.warn("PyTorch is required for loading Torchvision pre-trained weights.")

_TEMP_DIR = "/tmp/.eqx"
_Url = NewType("_Url", str)
MODEL_URLS = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


def load_torch_weights(
    model: eqx.Module, filepath: Path = None, url: "_Url" = None
) -> eqx.Module:
    """Loads weights from a PyTorch serialised file.

    ???+ warning

        This method requires installation of the [`torch`](https://pypi.org/project/torch/) package.

    !!! note

        - This function assumes that Eqxvision's ordering of class
          attributes mirrors the `torchvision.models` implementation.
        - The saved checkpoint should **only** contain model parameters as keys.

    **Arguments:**

    - model: An `eqx.Module` for which the `jnp.ndarray` leaves are
        replaced by corresponding `PyTorch` weights.
    - filepath: `Path` to the downloaded `PyTorch` model file.
    - url: `URL` for the `PyTorch` model file. The file is downloaded to `/tmp/.eqx/` folder.

    **Returns:**
        The model with weights loaded from the `PyTorch` checkpoint.
    """
    if filepath is None and url is None:
        raise ValueError("Both filepath and url cannot be empty!")
    elif filepath and url:
        warnings.warn(f"Overriding `url` with with filepath: {filepath}.")
        url = None
    if url:
        global _TEMP_DIR
        filepath = os.path.join(_TEMP_DIR, os.path.basename(url))
        if os.path.exists(filepath):
            logging.info(
                f"Downloaded file exists at f{filepath}. Using the cached file!"
            )
        else:
            os.makedirs(_TEMP_DIR, exist_ok=True)
            torch.hub.download_url_to_file(url, filepath)
    if not os.path.exists(filepath):
        raise ValueError(f"filepath: {filepath} does not exist!")

    weights = torch.load(filepath, map_location="cpu")
    weights_iterator = iter(
        [
            jnp.asarray(weight.detach().numpy())
            for name, weight in weights.items()
            if "bn.running" not in name and "bn.num" not in name
        ]
    )
    leaves, tree_def = jtu.tree_flatten(model)

    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and not (
            leaf.size == 1 and isinstance(leaf.item(), bool)
        ):
            new_weights = next(weights_iterator)
            new_leaves.append(jnp.reshape(new_weights, leaf.shape))
        else:
            new_leaves.append(leaf)

    return jtu.tree_unflatten(tree_def, new_leaves)

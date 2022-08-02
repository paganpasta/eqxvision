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
        [jnp.asarray(weight.detach().numpy()) for _, weight in weights.items()]
    )
    leaves, tree_def = jtu.tree_flatten(model)

    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray):
            new_weights = jnp.reshape(next(weights_iterator), leaf.shape)
            new_leaves.append(new_weights)
        else:
            new_leaves.append(leaf)

    return jtu.tree_unflatten(tree_def, new_leaves)

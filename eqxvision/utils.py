import logging
import os
import sys
import warnings
from typing import Any, Callable, NewType, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.custom_types import PyTree


try:
    import torch
except ImportError:
    warnings.warn("PyTorch is required for loading Torchvision pre-trained weights.")

_TEMP_DIR = "/tmp/.eqx"
_Url = NewType("_Url", str)

SEGMENTATION_URLS = {
    "fcn_resnet50": "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth"
}

CLASSIFICATION_URLS = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "convnext_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    "convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    "densenet161": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
    "efficientnet_v2_s": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
    "efficientnet_v2_m": "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
    "efficientnet_v2_l": "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-58fc7688.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-0d7bc02a.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-9180c971.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-8db6d4b5.pth",
    "regnet_y_128gf": "https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-2b70d774.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-ba3796d7.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-6eb8fdc6.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "shufflenetv2_x0.5": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenetv2_x1.0": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
    "swin_t": "https://download.pytorch.org/models/swin_t-704ceda3.pth",
    "swin_s": "https://download.pytorch.org/models/swin_s-5e29d889.pth",
    "sim_b": "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
    "swin_v2_t": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
    "swin_v2_s": "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
    "sim_v2_b": "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
    "vit_small_patch16_224_dino": "https://dl.fbaipublicfiles.com/dino/"
    "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    "vit_small_patch8_224_dino": "https://dl.fbaipublicfiles.com/dino/"
    "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    "vit_base_patch16_224_dino": "https://dl.fbaipublicfiles.com/dino/"
    "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "vit_base_patch8_224_dino": "https://dl.fbaipublicfiles.com/dino/"
    "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def load_torch_weights(
    model: eqx.Module,
    torch_weights: str = None,
) -> eqx.Module:
    """Loads weights from a PyTorch serialised file.

    ???+ warning

        This method requires installation of the [`torch`](https://pypi.org/project/torch/) package.

    !!! note

        - This function assumes that Eqxvision's ordering of class
          attributes mirrors the `torchvision.models` implementation.
        - The saved checkpoint should **only** contain model parameters as keys.

    **Arguments:**

    - `model`: An `eqx.Module` for which the `jnp.ndarray` leaves are
        replaced by corresponding `PyTorch` weights.
    - `weights`: A string either pointing to `PyTorch` weights on disk or the download `URL`.
    - `filepath`: `Path` to the downloaded `PyTorch` model file.
    - `url`: `URL` for the `PyTorch` model file. The file is downloaded to `/tmp/.eqx/` folder.

    **Returns:**
        The model with weights loaded from the `PyTorch` checkpoint.
    """
    if "torch" not in sys.modules:
        raise RuntimeError(
            " Torch package not found! Pretrained is only supported with the torch package."
        )

    if torch_weights is None:
        raise ValueError("torch_weights parameter cannot be empty!")

    if not os.path.exists(torch_weights):
        global _TEMP_DIR
        filepath = os.path.join(_TEMP_DIR, os.path.basename(torch_weights))
        if os.path.exists(filepath):
            logging.info(
                f"Downloaded file exists at f{filepath}. Using the cached file!"
            )
        else:
            os.makedirs(_TEMP_DIR, exist_ok=True)
            torch.hub.download_url_to_file(torch_weights, filepath)
    else:
        filepath = torch_weights
    saved_weights = torch.load(filepath, map_location="cpu")
    weights_iterator = iter(
        [
            (name, jnp.asarray(weight.detach().numpy()))
            for name, weight in saved_weights.items()
            if "running" not in name and "num_batches" not in name
        ]
    )

    bn_s = []
    for name, weight in saved_weights.items():
        if "running_mean" in name:
            bn_s.append(False)
            bn_s.append(jnp.asarray(weight.detach().numpy()))
        elif "running_var" in name:
            bn_s.append(jnp.asarray(weight.detach().numpy()))
    bn_iterator = iter(bn_s)

    leaves, tree_def = jtu.tree_flatten(model)

    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and not (
            leaf.size == 1 and isinstance(leaf.item(), bool)
        ):
            (weight_name, new_weights) = next(weights_iterator)
            new_leaves.append(jnp.reshape(new_weights, leaf.shape))
        else:
            new_leaves.append(leaf)

    model = jtu.tree_unflatten(tree_def, new_leaves)

    def set_experimental(iter_bn, x):
        def set_values(y):
            if isinstance(y, eqx.experimental.StateIndex):
                current_val = next(iter_bn)
                if isinstance(current_val, bool):
                    eqx.experimental.set_state(y, jnp.asarray(False))
                else:
                    running_mean, running_var = current_val, next(iter_bn)
                    eqx.experimental.set_state(y, (running_mean, running_var))
            return y

        return jtu.tree_map(
            set_values, x, is_leaf=lambda _: isinstance(_, eqx.experimental.StateIndex)
        )

    model = jtu.tree_map(set_experimental, bn_iterator, model)
    return model


class AuxData:
    """A simple container for aux data."""

    def __init__(self):
        self.data = None

    def update(self, x: Any):
        self.data = x


def _make_intermediate_layer_wrapper():
    aux = AuxData()

    class IntermediateWrapper(eqx.Module):
        layer: eqx.Module

        def __call__(self, x, *, key=None):
            out = self.layer(x, key=key)
            aux.update(out)
            return out

    return aux, IntermediateWrapper


def intermediate_layer_getter(
    model: PyTree, get_target_layers: Callable
) -> "eqx.Module":
    """Wraps intermediate layers of a model for accessing intermediate activations. Based on a discussion
    [here](https://github.com/patrick-kidger/equinox/issues/186).

    **Arguments:**

    - `model`: A PyTree representing the neural network model
    - `get_target_layers`: A callable function which returns a sequence
        of layers from the `model`.

    **Returns:**
    An `equinox.Module` which contains `model` with the layers of interest wrapped for storing intermediate outputs.
    """
    target_layers = get_target_layers(model)
    auxs, wrappers = zip(
        *[_make_intermediate_layer_wrapper() for _ in range(len(target_layers))]
    )
    model = eqx.tree_at(
        where=get_target_layers,
        pytree=model,
        replace=[
            wrapper(target_layer)
            for (wrapper, target_layer) in zip(wrappers, target_layers)
        ],
    )

    class IntermediateLayerGetter(eqx.Module):
        model: eqx.Module

        def __call__(self, x, *, key=None):
            out = self.model(x, key=key)
            return out, auxs

    return IntermediateLayerGetter(model)

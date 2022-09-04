from typing import Any, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.image as jim
import jax.random as jr
from equinox.custom_types import Array


class _SimpleSegmentationModel(eqx.Module):
    backbone: eqx.Module
    classifier: eqx.Module
    aux_classifier: eqx.Module

    def __init__(
        self,
        backbone: "eqx.Module",
        classifier: "eqx.Module",
        aux_classifier: Optional["eqx.Module"] = None,
    ) -> None:
        """

        **Arguments:**

        - `backbone`: the network used to compute the features for the model
            The backbone returns `embedding_features(Ignored)`, `[output features of intermediate layers]`.
        - `classifier`: module that takes last of the intermediate outputs from the
            backbone and returns a dense prediction
        - `aux_classifier`: If used, an auxiliary classifier similar to `classifier` for the auxiliary layer
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def __call__(
        self, x: Array, *, key: "jax.random.PRNGKey"
    ) -> Tuple[Union[Any, Array], Array]:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`

        **Returns:**
        A tuple with outputs from the intermediate layers.
        """
        keys = jr.split(key, 3)
        _, xs = self.backbone(x, key=keys[0])

        x_clf = self.classifier(xs[-1].data, key=keys[1])
        target_shape = (x_clf.shape[0], x.shape[-2], x.shape[-1])
        x_clf = jim.resize(x_clf, shape=target_shape, method="bilinear")

        if self.aux_classifier is not None:
            x_aux = self.aux_classifier(xs[0].data, key=keys[2])
            target_shape = (x_aux.shape[0], x.shape[-2], x.shape[-1])
            x_aux = jim.resize(x_aux, shape=target_shape, method="bilinear")
            return x_aux, x_clf

        return None, x_clf

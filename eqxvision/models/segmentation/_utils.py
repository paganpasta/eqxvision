from typing import Any, Optional, Tuple

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
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Tuple[Any, Any]:
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

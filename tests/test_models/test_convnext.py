import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestConvNext:
    def test_pretrained(self, getkey, demo_image, net_preds):
        keys = jax.random.split(getkey(), 1)
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.convnext_tiny(torchweights=CLASSIFICATION_URLS["alexnet"])
        model = eqx.tree_inference(model, True)
        eqx_outputs = forward(model, img, keys)
        pt_outputs = net_preds["convnext_tiny"]

        assert jnp.argmax(eqx_outputs) == jnp.argmax(pt_outputs)

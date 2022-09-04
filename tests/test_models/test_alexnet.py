import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestAlexNet:
    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.alexnet(torchweights=CLASSIFICATION_URLS["alexnet"])

        pt_outputs = net_preds["alexnet"]
        new_model = eqx.tree_inference(model, True)
        keys = jax.random.split(getkey(), 1)
        eqx_outputs = forward(new_model.features, img, keys)

        assert jnp.isclose(pt_outputs, eqx_outputs, atol=1e-4).all()

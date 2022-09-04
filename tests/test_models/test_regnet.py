import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestRegNet:
    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.regnet_x_400mf(
            torch_weights=CLASSIFICATION_URLS["regnet_x_400mf"]
        )
        model = eqx.tree_inference(model, True)
        eqx_outputs = forward(model, img, keys)
        pt_outputs = net_preds["regnet_x_400mf"]

        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()

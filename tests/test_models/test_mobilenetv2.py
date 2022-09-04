import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestMobileNetv2:
    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.mobilenet_v2(torch_weights=CLASSIFICATION_URLS["mobilenet_v2"])
        model = eqx.tree_inference(model, True)
        pt_outputs = net_preds["mobilenet_v2"]
        eqx_outputs = forward(model, img, keys)

        assert jnp.argmax(eqx_outputs, axis=1) == jnp.argmax(pt_outputs, axis=1)

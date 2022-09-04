import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestMobileNetv3:
    def test_pretrained(self, demo_image, net_preds, getkey):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.mobilenet_v3_small(
            torch_weights=CLASSIFICATION_URLS["mobilenet_v3_small"]
        )
        model = eqx.tree_inference(model, True)
        output = forward(model, img, getkey())

        pt_output = net_preds["mobilenet_v3_small"]
        assert jnp.isclose(output, pt_output, atol=1e-4).all()

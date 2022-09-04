import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision.utils import CLASSIFICATION_URLS


class TestShuffleNetV2:
    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.shufflenet_v2_x0_5(
            torch_weights=CLASSIFICATION_URLS["shufflenetv2_x0.5"]
        )
        model = eqx.tree_inference(model, True)
        pt_outputs = net_preds["shufflenetv2_x0.5"]
        eqx_outputs = forward(model, img, keys)

        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()

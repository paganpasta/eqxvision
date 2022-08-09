import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models
from eqxvision import utils


class TestAlexNet:
    answer = (1, 1000)

    def test_output_shape(self, getkey, demo_image):
        c_counter = 0

        @eqx.filter_jit
        def forward(model, x, key):
            nonlocal c_counter
            c_counter += 1
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(model)(x, key=keys)

        model = models.alexnet(num_classes=1000)
        output = forward(model, demo_image, getkey())
        assert output.shape == self.answer
        forward(model, demo_image, getkey())
        assert c_counter == 1

    def test_pretrained(self, getkey, demo_image, net_preds):
        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.alexnet(pretrained=False)
        new_model = utils.load_torch_weights(
            model=model, url=utils.MODEL_URLS["alexnet"]
        )

        new_model = eqx.tree_inference(new_model, True)
        assert model != new_model

        pt_outputs = net_preds["alexnet"]
        new_model = eqx.tree_inference(new_model, True)
        keys = jax.random.split(getkey(), 1)
        eqx_outputs = forward(new_model.features, demo_image, keys)

        assert jnp.isclose(pt_outputs, eqx_outputs, atol=1e-4).all()

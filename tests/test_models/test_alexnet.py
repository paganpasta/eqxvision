import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models


class TestAlexNet:
    answer = (1, 1000)

    def test_output_shape(self, getkey, demo_image):
        img = demo_image(224)
        c_counter = 0

        @eqx.filter_jit
        def forward(model, x, key):
            nonlocal c_counter
            c_counter += 1
            keys = jax.random.split(key, x.shape[0])
            return jax.vmap(model)(x, key=keys)

        model = models.alexnet(num_classes=1000)
        output = forward(model, img, getkey())
        assert output.shape == self.answer
        forward(model, img, getkey())
        assert c_counter == 1

    def test_pretrained(self, getkey, demo_image, net_preds):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.alexnet(pretrained=True)

        pt_outputs = net_preds["alexnet"]
        new_model = eqx.tree_inference(model, True)
        keys = jax.random.split(getkey(), 1)
        eqx_outputs = forward(new_model.features, img, keys)

        assert jnp.isclose(pt_outputs, eqx_outputs, atol=1e-4).all()

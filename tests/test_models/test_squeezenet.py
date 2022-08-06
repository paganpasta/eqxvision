import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import eqxvision.models as models


model_list = [models.squeezenet1_0]


class TestSqueezeNet:
    random_image = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224))
    answer = (1, 1000)

    @pytest.mark.parametrize("model_func", model_list)
    def test_sneNet(self, model_func, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = model_func(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

    def test_pretrained(self, getkey, demo_image, net_preds):
        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.squeezenet1_0(pretrained=True)

        new_model = eqx.tree_inference(model, True)

        pt_outputs = net_preds["squeezenet1_0"]
        new_model = eqx.tree_inference(new_model, True)
        keys = jax.random.split(getkey(), 1)
        eqx_outputs = forward(new_model, demo_image, keys)

        assert jnp.argmax(pt_outputs) == jnp.argmax(eqx_outputs)

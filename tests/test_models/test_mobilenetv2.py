import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import eqxvision.models as models


model_list = [("mobilenet_v2", models.mobilenet_v2)]


class TestMobileNetv2:
    answer = (1, 1000)

    @pytest.mark.parametrize("model_func", model_list)
    def test_mobilenet(self, model_func, demo_image, getkey):
        img = demo_image(224)

        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = model_func[1](num_classes=1000)
        output = forward(model, img, getkey())
        assert output.shape == self.answer

    @pytest.mark.parametrize("model_func", model_list)
    def test_pretrained(self, getkey, model_func, demo_image, net_preds):
        img = demo_image(224)
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = model_func[1](pretrained=True)
        model = eqx.tree_inference(model, True)
        pt_outputs = net_preds[model_func[0]]
        eqx_outputs = forward(model, img, keys)

        assert jnp.argmax(eqx_outputs, axis=1) == jnp.argmax(pt_outputs, axis=1)

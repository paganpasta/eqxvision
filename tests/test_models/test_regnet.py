import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import eqxvision.models as models


model_list = [("regnet_y_400mf", models.regnet_y_400mf)]


class TestRegNet:
    answer = (1, 1000)

    @pytest.mark.parametrize("model_func", model_list)
    def test_resnets(self, model_func, demo_image, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = model_func[1](num_classes=1000)
        output = forward(model, demo_image, getkey())
        assert output.shape == self.answer

    @pytest.mark.parametrize("model_func", model_list)
    def test_pretrained(self, getkey, model_func, demo_image):
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = model_func[1](pretrained=True)
        model = eqx.tree_inference(model, True)
        eqx_outputs = forward(model, demo_image, keys)

        import numpy
        import torch
        import torchvision

        m = torchvision.models.regnet_y_400mf(
            weights=torchvision.models.RegNet_Y_400MF_Weights.IMAGENET1K_V1
        )
        m.eval()
        with torch.no_grad():
            pt_outputs = m(torch.tensor(numpy.asarray(demo_image))).numpy()
        # pt_outputs = net_preds[model_func[0]]
        assert jnp.isclose(eqx_outputs, pt_outputs, atol=1e-4).all()

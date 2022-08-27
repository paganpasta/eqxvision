import equinox as eqx
import jax
import jax.numpy as jnp

import eqxvision.models as models


class TestSwinTransformer:
    answer = (1, 1000)

    def test_swin_transformer(self, demo_image, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.swin_t()
        output = forward(model, demo_image(224), getkey())
        assert output.shape == self.answer

    def test_swin_transformer_v2(self, demo_image, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.swin_v2_t()
        output = forward(model, demo_image(256), getkey())
        assert output.shape == self.answer

    def test_pretrained_swin(self, getkey, demo_image, net_preds):
        keys = jax.random.split(getkey(), 1)

        @eqx.filter_jit
        def forward(net, imgs, keys):
            outputs = jax.vmap(net, axis_name="batch")(imgs, key=keys)
            return outputs

        model = models.swin_t(pretrained=True)
        model = eqx.tree_inference(model, True)
        eqx_outputs = forward(model, demo_image(224), keys)

        pt_outputs = net_preds["swin_t"]

        assert jnp.argmax(eqx_outputs) == jnp.argmax(pt_outputs)

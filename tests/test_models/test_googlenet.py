import equinox as eqx
import jax

import eqxvision.models as models


class TestGoogLeNet:
    answer = (1, 1000)

    def test_output_shape(self, demo_image, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.googlenet(num_classes=1000, aux_logits=False)
        output = forward(model, demo_image, getkey())
        assert output.shape == self.answer

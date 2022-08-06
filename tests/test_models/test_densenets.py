import equinox as eqx
import jax
import pytest

import eqxvision.models as models


model_list = [models.densenet121]


class TestDenseNet:
    random_image = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224))
    answer = (1, 1000)

    @pytest.mark.parametrize("model_func", model_list)
    def test_densenets(self, model_func, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = model_func(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

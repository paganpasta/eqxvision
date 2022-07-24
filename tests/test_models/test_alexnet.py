import pytest
import eqxvision.models as models
import jax
import equinox as eqx


class TestAlexNet:

    def test_output_shape(self, getkey):
        random_image = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224))

        @eqx.filter_jit
        def forward(model, x, key):
           keys = jax.random.split(key, x.shape[0])
           return jax.vmap(model)(x, key=keys)

        model = models.alexnet(num_classes=1000)
        output = forward(model, random_image, getkey())
        answer = (1, 1000)
        assert output.shape == answer


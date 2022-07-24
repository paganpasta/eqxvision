import pytest
import eqxvision.models as models
import jax
import equinox as eqx


class TestResNet:
    random_image = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(1, 3, 224, 224))
    answer = (1, 1000)

    def test_resnet(self, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.resnet18(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.resnet34(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.resnet50(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.resnet101(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.resnet152(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

    def test_wide_resnet(self, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x)
            return ans

        model = models.wide_resnet50_2(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.wide_resnet101_2(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

    def test_resnetxd(self, getkey):
        @eqx.filter_jit
        def forward(net, x, key):
            keys = jax.random.split(key, x.shape[0])
            ans = jax.vmap(net, axis_name="batch")(x, key=keys)
            return ans

        model = models.resnext50_32x4d(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer

        model = models.resnext101_32x8d(num_classes=1000)
        output = forward(model, self.random_image, getkey())
        assert output.shape == self.answer


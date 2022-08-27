import equinox as eqx
import jax
import jax.random as jr
import pytest

import eqxvision.models as models


class TestVit:
    def test_vit_attention(self, getkey):
        c_counter = 0

        @eqx.filter_jit
        def forward(net, x, keys):
            nonlocal c_counter
            c_counter += 1
            return jax.vmap(net)(x, key=keys)

        random_input = jax.random.uniform(key=getkey(), shape=(1, 8, 32))
        answer, answer_attn = (1, 8, 32), (1, 1, 4, 8, 8)
        net = models._VitAttention(
            32, num_heads=4, qkv_bias=True, attn_drop=0.2, proj_drop=0.2, key=getkey()
        )
        keys = jax.random.split(getkey(), random_input.shape[0])

        output, attn = forward(net, random_input, keys)
        assert output.shape == answer
        assert attn.shape == answer_attn

        random_input = jax.random.uniform(key=getkey(), shape=(1, 8, 32))
        output, attn = forward(net, random_input, keys)
        assert output.shape == answer
        assert attn.shape == answer_attn
        assert c_counter == 1

    def test_vit_block(self, getkey):
        c_counter = 0

        @eqx.filter_jit
        def forward(net, x, keys, attn=False):
            nonlocal c_counter
            c_counter += 1
            return eqx.filter_vmap(net)(x, return_attention=attn, key=keys)

        random_input = jax.random.uniform(key=getkey(), shape=(1, 8, 32))
        answer, answer_attn = (1, 8, 32), (1, 1, 4, 8, 8)
        net = models._VitBlock(32, num_heads=4, key=getkey())
        keys = jax.random.split(getkey(), random_input.shape[0])

        output = forward(net, random_input, keys)
        assert output.shape == answer

        random_input = jax.random.uniform(key=getkey(), shape=(1, 8, 32))
        forward(net, random_input, keys)
        assert c_counter == 1

        attn = forward(net, random_input, keys, attn=True)
        assert attn.shape == answer_attn
        assert c_counter == 2

    def test_vit_self_attention(self, getkey):
        c_counter = 0

        @eqx.filter_jit
        def forward(net, x, keys):
            nonlocal c_counter
            c_counter += 1
            return jax.vmap(net.get_last_self_attention)(x, key=keys)

        random_input = jax.random.uniform(key=getkey(), shape=(1, 3, 224, 224))
        answer = (1, 1, 12, 197, 197)
        net = models.VisionTransformer(img_size=224, patch_size=16)
        keys = jax.random.split(getkey(), random_input.shape[0])
        with pytest.raises(ValueError):
            forward(net, random_input, keys)

        net = eqx.tree_inference(net, True)
        random_input = jax.random.uniform(key=getkey(), shape=(1, 3, 224, 224))
        output = forward(net, random_input, keys)
        assert output.shape == answer
        assert c_counter == 2

    @pytest.mark.parametrize(
        "model_func", [models.vit_tiny, models.vit_small, models.vit_base]
    )
    def test_vit_variants(self, model_func, getkey):
        c_counter = 0

        @eqx.filter_jit
        def forward(net, x, keys):
            nonlocal c_counter
            c_counter += 1
            return jax.vmap(net)(x, key=keys)

        random_input = jax.random.uniform(key=getkey(), shape=(1, 3, 224, 224))
        answer = (1, 1000)
        net = model_func(num_classes=1000)
        keys = jax.random.split(getkey(), random_input.shape[0])

        output = forward(net, random_input, keys)
        assert output.shape == answer

    def test_vit_pretrained(self, demo_image, getkey):
        img = demo_image(224)
        net = models.vit_small(num_classes=0, pretrained=True)
        keys = jr.split(getkey(), 1)
        output = jax.vmap(net)(img, key=keys)

        assert output.shape == (1, 384)

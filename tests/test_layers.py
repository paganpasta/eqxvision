import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom

import eqxvision.layers as layers


def test_chain(getkey):
    c_counter = 0

    @eqx.filter_jit
    def forward(net, xs, keys):
        nonlocal c_counter
        c_counter += 1
        return jax.vmap(net)(xs, key=keys)

    chain = eqx.nn.Sequential(
        [
            eqx.nn.Identity(),
            layers.Activation(jnn.relu),
        ]
    )
    x = jnp.array([[-1, -2, -3], [1, 2, 3]])
    keys = jrandom.split(getkey(), 2)
    output = forward(chain, x, keys)
    assert output.shape == (2, 3)
    assert (output >= 0).all()

    forward(chain, x, keys)
    assert c_counter == 1


def test_patch_embed(getkey):
    @eqx.filter_jit
    def forward(net, xs):
        return jax.vmap(net)(xs)

    patch_embed = layers.PatchEmbed(norm_layer=eqx.nn.LayerNorm)
    x = jrandom.uniform(getkey(), (1, 3, 224, 224))
    output = forward(patch_embed, x)
    assert output.shape == (1, 196, 768)


def test_mlp_proj(getkey):
    @eqx.filter_jit
    def forward(net, xs, keys):
        return jax.vmap(net)(xs, key=keys)

    mlp = layers.MlpProjection(
        in_features=20,
        out_features=10,
        hidden_features=5,
        act_layer=jnn.gelu,
        key=getkey(),
    )
    x = jrandom.uniform(getkey(), (10, 20))
    output = forward(mlp, x, jrandom.split(getkey(), 10))
    assert output.shape == (10, 10)


def test_drop_path(getkey):
    @eqx.filter_jit
    def forward(net, xs, keys):
        return jax.vmap(net)(xs, key=keys)

    dp = layers.DropPath(p=0.5)
    x = jrandom.uniform(getkey(), (10, 20), minval=1, maxval=10)
    output = forward(dp, x, jrandom.split(getkey(), 10))
    assert output.shape == (10, 20)
    assert (output == 0).any()

    dp_eval = eqx.tree_inference(dp, True)
    output = forward(dp_eval, x, jrandom.split(getkey(), 10))
    assert output.shape == (10, 20)
    assert (output != 0).all()

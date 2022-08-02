import random

import jax.numpy as jnp
import jax.random as jrandom
import pytest


@pytest.fixture()
def getkey():
    def _getkey():
        ii32 = jnp.iinfo(jnp.int32)
        return jrandom.PRNGKey(random.randint(0, ii32.max - 1))

    return _getkey


@pytest.fixture()
def random_image(input_shape):
    return jrandom.uniform(key=jrandom.PRNGKey(0), shape=input_shape)

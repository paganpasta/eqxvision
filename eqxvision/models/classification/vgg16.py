import jax
import jax.numpy as jnp
from equinox import Module
from equinox.nn import MaxPool2D, Linear, Conv2d


def preprocess_input(image, mean=[103.939, 116.779, 123.68]):
    image = image.astype(jnp.float32)
    image = image[..., ::-1]
    image = jnp.moveaxis(image, 2, 0)
    mean = jnp.array(mean)
    mean = jnp.expand_dims(mean, axis=[1, 2])
    return image - mean


def flatten(x):
    x = jnp.moveaxis(x, 0, 2)
    x = jnp.reshape(x, -1)
    return x


def build_block(intro_channels, outro_channels, key):
    keys = jax.random.split(key, len(outro_channels))
    block = []
    for outro_channel, key in zip(outro_channels, keys):
        conv2D = Conv2d(intro_channels, outro_channel, 3, padding=1, key=key)
        block.extend([conv2D, jax.nn.relu])
        intro_channels = outro_channel
    block.append(MaxPool2D((2, 2), stride=(2, 2)))
    return block


def build_head(key, in_features, num_classes):
    key_1, key_2, key_3 = jax.random.split(key, 3)
    return [flatten,
            Linear(in_features, 4096, key=key_1),
            jax.nn.relu,
            Linear(4096, 4096, key=key_2),
            jax.nn.relu,
            Linear(4096, num_classes, key=key_3),
            jax.nn.softmax]


class VGG16(Module):
    layers: list

    def __init__(self, key, num_classes=1000):
        key_1, key_2, key_3, key_4, key_5, key_6 = jax.random.split(key, 6)
        block_1 = build_block(3, [64, 64], key_1)
        block_2 = build_block(64, [128, 128], key_2)
        block_3 = build_block(128, [256, 256, 256], key_3)
        block_4 = build_block(256, [512, 512, 512], key_4)
        block_5 = build_block(512, [512, 512, 512], key_5)
        head = build_head(key_6, 512 * 7 * 7, num_classes)
        self.layers = block_1 + block_2 + block_3 + block_4 + block_5 + head

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

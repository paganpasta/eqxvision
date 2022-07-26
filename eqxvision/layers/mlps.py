from typing import Callable, Tuple, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
from equinox.custom_types import Array


class MlpProjection(eqx.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    fc1: nn.Linear
    act: Callable
    drop1: nn.Dropout
    fc2: nn.Linear
    drop2: nn.Dropout

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: Callable = None,
        drop: Union[float, Tuple[float]] = 0.0,
        *,
        key: "jax.random.PRNGKey" = None
    ):
        """**Arguments:**

        - in_features: The expected dimension of the input.
        - hidden_features: Dimensionality of the hidden layer.
        - out_features: The dimension of the output feature.
        - act_layer: Activation function to be applied to the intermediate layers.
        - drop: The probability associated with `Dropout`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
        keys = jrandom.split(key, 2)
        self.fc1 = nn.Linear(in_features, hidden_features, key=keys[0])
        self.act = act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, key=keys[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array.
        - `key`: Utilised by few layers in the network such as `Dropout` or `BatchNorm`.
        """
        keys = jrandom.split(key, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, key=keys[0])
        x = self.fc2(x)
        x = self.drop2(x, key=keys[1])
        return x

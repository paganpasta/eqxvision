from typing import Any, Callable, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.random as jrandom


class Chain(eqx.Module):
    """Similar to `equinox.nn.Sequence` but also allows callables in the sequence."""

    layers: Sequence[Union[eqx.Module, Callable]]

    def __call__(self, x: Any, *, key: Optional["jax.random.PRNGKey"] = None) -> Any:
        """**Arguments:**

        - `x`: Argument passed to the first member of the sequence.
        - `key`: A `jax.random.PRNGKey`, which will be split and passed to every layer
            to provide any desired randomness. (Optional. Keyword only argument.)
        **Returns:**

        The output of the last member of the sequence.
        """

        if key is None:
            keys = [None] * len(self.layers)
        else:
            keys = jrandom.split(key, len(self.layers))
        for layer, key in zip(self.layers, keys):
            if isinstance(layer, eqx.Module):
                x = layer(x, key=key)
            else:
                x = layer(x)
        return x

    def __getitem__(self, i: Union[int, slice]) -> eqx.Module:
        if isinstance(i, int):
            return self.layers[i]
        elif isinstance(i, slice):
            return Chain(self.layers[i])
        else:
            raise TypeError(f"Indexing with type {type(i)} is not supported")


Chain.__init__.__doc__ = """**Arguments:**
- `layers`: A sequence of `equinox.Module`s and/or Callables.
"""

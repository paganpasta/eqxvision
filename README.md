# Eqxvision
[![PyPI](https://img.shields.io/pypi/v/eqxvision?style=flat-square)](https://pypi.org/project/eqxvision/) 
[![Github](https://img.shields.io/badge/Documentation-link-yellowgreen)](https://eqxvision.readthedocs.io/en/latest/)
[![GitHub Release Date](https://img.shields.io/github/release-date/paganpasta/eqxvision?style=flat-square)](https://github.com/paganpasta/eqxvision/releases) 
[![GitHub](https://img.shields.io/github/license/paganpasta/eqxvision?style=flat-square)](https://github.com/paganpasta/eqxvision/blob/main/LICENSE.md)

Eqxvision is a package of popular computer vision model architectures built using [Equinox](https://docs.kidger.site/equinox/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install eqxvision.

```bash
pip install eqxvision
```

*requires:* `python>=3.7`

*optional:* `torch`, only if `pretrained` models are required. 

## Documentation

Available at [https://eqxvision.readthedocs.io/en/latest/](https://eqxvision.readthedocs.io/en/latest/).

## Usage

Picking a model and doing a forward pass is as simple as ...

```python
    import jax
    import jax.random as jr
    import equinox as eqx
    from eqxvision.models import alexnet
    
    @eqx.filter_jit
    def forward(net, images, key):
        keys = jax.random.split(key, images.shape[0])
        output = jax.vmap(net, axis_name=('batch'))(images, key=keys)
        ...
        
    net = alexnet(num_classes=1000)
    
    images = jr.uniform(jr.PRNGKey(0), shape=(1,3,224,224))
    output = forward(net, images, jr.PRNGKey(0))
```

## What's New?
- All `torchvision` classification models are now ported to `eqxvision`!

    !!! note
        Only `DINO` pretrained weights are available for `ViTs`.


## Get Started!

Start with any one of these easy to follow [tutorials](https://eqxvision.readthedocs.io/en/latest/getting_started/Transfer_Learning/). 
       
       
## Tips
- Better to use `@equinox.jit_filter` instead of `@jax.jit`
- Advisable to use `jax.{v,p}map` with `axis_name='batch'` for all models
- Don't forget to switch to `inference` mode for evaluations
- Wrap with `eqx.filter(net, eqx.is_array)` for `Optax` initialisation.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Development Process
If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `dev`.
2. If you have modified the code (new feature or bug-fix), please add unit tests.
3. If you have changed APIs, update the documentation. Make sure the documentation builds. `mkdocs serve`
4. Ensure the test suite passes. `pytest tests -vvv`
5. Make sure your code passes the formatting checks. Automatically checked with a `pre-commit` hook. 


## Acknowledgements
- [Equinox](https://github.com/patrick-kidger/equinox)
- [Patrick Kidger](https://github.com/patrick-kidger)
- [Torchvision](https://pytorch.org/vision/stable/index.html)

## License
[MIT](https://choosealicense.com/licenses/mit/)

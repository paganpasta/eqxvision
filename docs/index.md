# Eqxvision

Eqxvision is a package of popular computer vision model architectures built using [Equinox](https://docs.kidger.site/equinox/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install eqxvision.

```bash
pip install eqxvision
```

*requires:* `python>=3.7`

## Usage
???+ Example
    Importing and doing a forward pass is as simple as
        ```python
        import jax
        import jax.random as jr
        import equinox as eqx
        from eqxvision.models import alexnet
        from eqxvision.utils import CLASSIFICATION_URLS
        
        
        @eqx.filter_jit
        def forward(net, images, key):
            keys = jax.random.split(key, images.shape[0])
            output = jax.vmap(net, axis_name=('batch'))(images, key=keys)
            ...
            
        net = alexnet(torch_weights=CLASSIFICATION_URLS['alexnet'])
        
        images = jr.uniform(jr.PRNGKey(0), shape=(1,3,224,224))
        output = forward(net, images, jr.PRNGKey(0))
        ```

## What's New?
- Backward incompatible changes to `v0.2.0` for loading a `pretrained` model.
- `FCN` and `DeepLabV3` segmentation models are now supported (checkout the [tutorial](getting_started/FCN_Segmentation.ipynb)).
- Almost all image classification models are ported from `torchvision`.
- New tutorial for generating [adversarial examples](getting_started/Adversarial_Attack.ipynb) and others coming soon.


## Get Started!

Start with any one of these easy to follow [tutorials](getting_started/Transfer_Learning.ipynb). 

## Tips
- Better to use `@equinox.filter_jit` instead of `@jax.jit`.
- Use `jax.{v,p}map` with `axis_name='batch'` when using models that use batch normalisation.
- Don't forget to switch to `inference` mode for evaluations. (`model = eqx.tree_inference(model)`)
- Initialise Optax optimisers as `optim.init(eqx.filter(net, eqx.is_array))`. ([See here.](https://docs.kidger.site/equinox/faq/#optax-is-throwing-an-error))




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
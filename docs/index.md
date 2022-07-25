# Eqxvision

Eqxvision is a package of popular computer vision model architectures built using [Equinox](https://docs.kidger.site/equinox/).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install eqxvision.

```bash
pip install eqxvision
```

## Usage

```python title="example.py"
import jax
import jax.random as jr
import equinox as eqx
from eqxvision.models import resnet18

@eqx.filter_jit
def forward(net, images, key):
    keys = jax.random.split(key, images.shape[0])
    jax.vmap(net)(images, key=keys)

net = resnet18(num_classes=1000)

images = jr.uniform(jr.PRNGKey(0), shape=(1,3,224,224))
output = forward(net, images, jr.PRNGKey(0))
```

## Tips
- Use `jax.vmap(net, axis_name='batch')(images)` for models with `batchnorms`.
- Don't forget to call `eqx.inference` for switching to `inference` mode.

## Roadmap

- [ ] Add VGGs, Inception, GoogLeNet
- [ ] Add/Explore functionality to load weights directly from torch.pth
- [ ] Doc fixes
- [ ] Build fixes
- [ ] Pre-commit Hooks


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgements
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Equinox](https://github.com/patrick-kidger/equinox)
- [Patrick Kidger](https://github.com/patrick-kidger)

## License
[MIT](https://choosealicense.com/licenses/mit/)
import random

import jax.numpy as jnp
import jax.random as jrandom
import pytest
import torch
from PIL import Image
from torchvision.transforms import transforms


@pytest.fixture()
def getkey():
    def _getkey():
        ii32 = jnp.iinfo(jnp.int32)
        return jrandom.PRNGKey(random.randint(0, ii32.max - 1))

    return _getkey


@pytest.fixture()
def img_transform():
    def _transform(img_size):
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    return _transform


@pytest.fixture()
def demo_image(img_transform):
    def _demo_image(img_size):
        img = Image.open("./tests/static/img.png")
        img = img.convert("RGB")
        return jnp.asarray(img_transform(img_size)(img).unsqueeze(0))

    return _demo_image


@pytest.fixture(scope="session")
def net_preds():
    gt_dicts = {}

    ckpt = torch.load("./tests/static/alexnet.pred.pth")
    gt_dicts["alexnet"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/convnext_tiny.pred.pth")
    gt_dicts["convnext_tiny"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/densenet121.pred.pth")
    gt_dicts["densenet121"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/efficientnet_b0.pred.pth")
    gt_dicts["efficientnet_b0"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/efficientnet_v2_s.pred.pth")
    gt_dicts["efficientnet_v2_s"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/googlenet.pred.pth")
    gt_dicts["googlenet"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/mobilenet_v2.pred.pth")
    gt_dicts["mobilenet_v2"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/mobilenet_v3_small.pred.pth")
    gt_dicts["mobilenet_v3_small"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/regnet_x_400mf.pred.pth")
    gt_dicts["regnet_x_400mf"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/resnet18.pred.pth")
    gt_dicts["resnet18"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/shufflenet_v2_x0_5.pred.pth")
    gt_dicts["shufflenetv2_x0.5"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/squeezenet1_0.pred.pth")
    gt_dicts["squeezenet1_0"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/swin_t.pred.pth")
    gt_dicts["swin_t"] = ckpt.detach().numpy()

    ckpt = torch.load("./tests/static/vgg11.pred.pth")
    gt_dicts["vgg11"] = ckpt["output"].detach().numpy()

    ckpt = torch.load("./tests/static/vgg11_bn.pred.pth")
    gt_dicts["vgg11_bn"] = ckpt["output"].detach().numpy()

    return gt_dicts

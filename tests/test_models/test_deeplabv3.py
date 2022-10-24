import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from eqxvision.models import deeplabv3, resnet50
from eqxvision.utils import SEGMENTATION_URLS


@eqx.filter_jit
def forward(model, x, key):
    aux, clf = jax.vmap(model, axis_name="batch")(x, key=key)
    return aux, clf


def test_deeplabv3(demo_image, net_preds):
    img = demo_image(224)
    net = deeplabv3(
        backbone=resnet50(replace_stride_with_dilation=[False, True, True]),
        intermediate_layers=lambda x: [x.layer3, x.layer4],
        aux_in_channels=1024,
        torch_weights=SEGMENTATION_URLS["deeplabv3_resnet50"],
    )
    net = eqx.tree_inference(net, True)
    aux, out = forward(net, img, key=jr.split(jr.PRNGKey(0), 1))

    pt_outputs = net_preds["deeplabv3_resnet50"]
    assert jnp.isclose(out, pt_outputs, atol=1e-4).all()

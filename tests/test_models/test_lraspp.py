import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from eqxvision.models.classification.mobilenetv3 import mobilenet_v3_large
from eqxvision.models.segmentation.lraspp import lraspp_mobilenet_v3_large
from eqxvision.utils import SEGMENTATION_URLS


@eqx.filter_jit
def forward(model, x, key):
    aux, clf = jax.vmap(model, axis_name="batch")(x, key=key)
    return aux, clf


def test_lraspp(demo_image, net_preds):
    img = demo_image(224)

    net = lraspp_mobilenet_v3_large(
        backbone=mobilenet_v3_large(dilated=True),
        intermediate_layers=lambda x: [4, 16],
        torch_weights=SEGMENTATION_URLS["lraspp_mobilenetv3_large"],
    )
    net = eqx.tree_inference(net, True)
    aux, out = forward(net, img, key=jr.split(jr.PRNGKey(0), 1))

    pt_outputs = net_preds["lraspp_mobilenetv3_large"]
    assert jnp.isclose(out, pt_outputs, atol=1e-4).all()

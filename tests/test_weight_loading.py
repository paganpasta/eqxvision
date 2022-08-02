import eqxvision.models as models
import eqxvision.utils as utils


def test_load_weights(getkey):
    model = models.alexnet(num_classes=1000)
    new_model = utils.load_torch_weights(
        model=model, url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    )
    assert model != new_model

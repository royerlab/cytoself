import torch

from ..vanilla import VanillaAE


def test_VanillaAE():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (64, 4, 4)
    model = VanillaAE(input_shape, emb_shape, input_shape)
    model.to(device)
    input_data = torch.randn((1,) + input_shape).to(device)
    out = model(input_data)
    assert out.shape == input_data.shape

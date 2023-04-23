import torch

from cytoself.trainer.autoencoder.base import BaseAE
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0


def test_VanillaAE():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (64, 4, 4)
    model = BaseAE(emb_shape, input_shape, input_shape)
    model.to(device)
    input_data = torch.randn((1,) + input_shape).to(device)
    out = model(input_data)
    assert out.shape == input_data.shape


def test_VanillaAE_custom():
    input_shape, emb_shape = (2, 100, 100), (64, 4, 4)
    encoder = efficientenc_b0(in_channels=input_shape[0], out_channels=emb_shape[0])
    decoder = DecoderResnet(input_shape=emb_shape, output_shape=input_shape)
    model = BaseAE(encoder=encoder, decoder=decoder)
    assert model.encoder == encoder
    assert model.decoder == decoder

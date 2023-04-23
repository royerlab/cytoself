import torch
from torch import nn
from torchvision.ops.misc import ConvNormActivation

block_args = [
    {
        'expand_ratio': 1,
        'kernel': 3,
        'stride': 1,
        'input_channels': 32,
        'out_channels': 16,
        'num_layers': 1,
    },
    {
        'expand_ratio': 6,
        'kernel': 3,
        'stride': 2,
        'input_channels': 16,
        'out_channels': 24,
        'num_layers': 2,
    },
]


def test_efficientenc_b0_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0

    model = efficientenc_b0()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b0(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    out = model(torch.randn((1, 32, 100, 100)))
    assert tuple(out.shape) == (1, 24, 50, 50)


def test_efficientenc_b0_channels():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0

    model = efficientenc_b0(block_args, in_channels=2, out_channels=64)
    assert len(model.features._modules) == len(block_args) + 2
    assert isinstance(model.features[0], ConvNormActivation)
    assert isinstance(model.features[-1], ConvNormActivation)
    assert model.features[0][0].in_channels == 2
    assert model.features[-1][0].out_channels == 64


def test_efficientenc_b1_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b1

    model = efficientenc_b1()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b1(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b2_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b2

    model = efficientenc_b2()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b2(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b3_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b3

    model = efficientenc_b3()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b3(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b4_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b4

    model = efficientenc_b4()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b4(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b5_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b5

    model = efficientenc_b5()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b5(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b6_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b6

    model = efficientenc_b6()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b6(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)


def test_efficientenc_b7_default():
    from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b7

    model = efficientenc_b7()
    assert len(model.features._modules) == 7
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)
    model = efficientenc_b7(block_args)
    assert len(model.features._modules) == len(block_args)
    assert isinstance(model.features[0], nn.Sequential)
    assert isinstance(model.features[-1], nn.Sequential)

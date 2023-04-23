import torch
from torch import nn

from cytoself.components.blocks.conv_block import Conv2dBN
from cytoself.components.blocks.residual_block import ResidualBlockRepeat
from cytoself.components.blocks.test.test_conv_block import test_Conv2dBN
from cytoself.components.blocks.test.test_residual_block import test_ResidualBlockRepeat
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet


def test_DecoderResnet():
    input_shape, output_shape, num_hiddens = (576, 4, 4), (64, 25, 25), 64

    model = DecoderResnet(input_shape, output_shape)
    assert not hasattr(model.decoder, 'dec_first')
    assert list(model.decoder)[0] == 'up1'
    assert len(list(model.decoder)) == 10

    model = DecoderResnet(input_shape, output_shape, num_hiddens=num_hiddens)
    data = torch.randn(1, 576, 4, 4)
    output = model(data)
    assert hasattr(model.decoder, 'dec_first')
    assert isinstance(model.decoder.dec_first, Conv2dBN)
    assert tuple(output.shape)[1:] == output_shape
    assert len(model.decoder._modules) == 11

    for key, val in {
        'dec_first': Conv2dBN,
        'up1': nn.Upsample,
        'resrep1': ResidualBlockRepeat,
        'resrep1last': Conv2dBN,
        'up2': nn.Upsample,
        'resrep2': ResidualBlockRepeat,
        'resrep2last': Conv2dBN,
        'up3': nn.Upsample,
        'resrep3': ResidualBlockRepeat,
        'resrep3last': Conv2dBN,
    }.items():
        assert hasattr(model.decoder, key)
        assert isinstance(getattr(model.decoder, key), val)
        if key == 'resrep1last':
            test_Conv2dBN(getattr(model.decoder, key))
        elif key == 'resrep1':
            test_ResidualBlockRepeat(getattr(model.decoder, key))
        elif key == 'resrep3last':
            last_layer = getattr(model.decoder, key)
            assert isinstance(last_layer.conv, nn.Conv2d)

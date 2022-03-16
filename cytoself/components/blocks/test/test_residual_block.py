from torch import nn
from cytoself.components.blocks.residual_block import ResidualBlockUnit2d, ResidualBlockRepeat
from cytoself.components.blocks.conv_block import Conv2dBN
from cytoself.components.blocks.test.test_conv_block import test_Conv2dBN


def test_ResidualBlockUnit2d(model=None):
    if model is None:
        model = ResidualBlockUnit2d(32)
    for key, val in {'conv1': Conv2dBN, 'conv2': nn.Conv2d, 'bn2': nn.BatchNorm2d, 'act2': nn.SiLU}.items():
        if hasattr(model, key):
            assert isinstance(getattr(model, key), val)
            if key == 'conv1':
                test_Conv2dBN(getattr(model, key))
        else:
            raise AttributeError('ResidualBlockUnit2d has no attribute ', key)


def test_ResidualBlockRepeat(model=None, n=2):
    if model is None:
        model = ResidualBlockRepeat(32, n)
    assert len(model.res_repeat._modules) == n
    for i in range(n):
        assert isinstance(getattr(model.res_repeat, f'res{i + 1}'), ResidualBlockUnit2d)
    test_ResidualBlockUnit2d(getattr(model.res_repeat, f'res{i + 1}'))

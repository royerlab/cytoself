from torch import nn
from cytoself.components.blocks.conv_block import calc_groups, Conv2dBN


def test_calc_groups():
    assert calc_groups(32, 32, verbose=False) == 32
    for i in (16, 2, 1):
        assert calc_groups(32, i, verbose=False) == i
        assert calc_groups(i, 32, verbose=False) == 1
    for i in (30, 15):
        assert calc_groups(32, i, verbose=False) == 1
        assert calc_groups(i, 32, verbose=False) == 1


def test_Conv2dBN():
    model = Conv2dBN(32, 16)
    for key, val in {'conv': nn.Conv2d, 'bn': nn.BatchNorm2d, 'act': nn.SiLU}.items():
        if hasattr(model, key):
            assert isinstance(model.conv, val)
        else:
            raise AttributeError('Conv2dBN has no attribute ', key)

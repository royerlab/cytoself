import pytest
from torch import nn

from cytoself.components.blocks.conv_block import Conv2dBN, calc_groups


def test_calc_groups():
    assert calc_groups(32, 32, verbose=False) == 32
    for i in (16, 2, 1):
        assert calc_groups(32, i, verbose=False) == i
        assert calc_groups(i, 32, verbose=False) == 1
    with pytest.warns(UserWarning):
        for i in (30, 14):
            assert calc_groups(32, i, verbose=True) == 1
            assert calc_groups(i, 32, verbose=True) == 1


def test_Conv2dBN(model=None):
    if model is None:
        model = Conv2dBN(32, 16)
    for key, val in {'conv': nn.Conv2d, 'bn': nn.BatchNorm2d, 'act': nn.SiLU}.items():
        assert hasattr(model, key)
        assert isinstance(getattr(model, key), val)


def test_Conv2dBN_conv_gp():
    model = Conv2dBN(32, 16, conv_gp='depthwise')
    assert model.conv.groups == 16
    model = Conv2dBN(32, 16, conv_gp=4)
    assert model.conv.groups == 4


def test_Conv2dBN_bn_affine():
    model = Conv2dBN(32, 16, bn_affine=True)
    assert model.bn.affine

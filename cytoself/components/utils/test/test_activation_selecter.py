import pytest
from torch import nn

from cytoself.components.utils.activation_selecter import act_layer


def test_act_layer():
    for key, val in {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'swish': nn.SiLU,
        'hswish': nn.Hardswish,
        'mish': nn.Mish,
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax,
    }.items():
        act = act_layer(key)
        assert isinstance(act, val)

    with pytest.raises(ValueError):
        act_layer('unknown')

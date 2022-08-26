from torch import nn

_ACT_DICT = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    'hswish': nn.Hardswish,
    'mish': nn.Mish,
    'sigmoid': nn.Sigmoid,
    'logsigmoid': nn.LogSigmoid,
    'softmax': nn.Softmax,
    'logsoftmax': nn.LogSoftmax,
}


def act_layer(act: str, **kwargs) -> nn.Module:
    """
    Selects activation layer

    Parameters
    ----------
    act : str
        Activation name

    Returns
    -------
    Activation module

    """
    act = act.lower()
    if act in _ACT_DICT:
        return _ACT_DICT[act](**kwargs)
    else:
        raise ValueError(f'{act} is not implemented.')

from torch import nn


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
    if act == 'relu':
        return nn.ReLU(**kwargs)
    elif act == 'lrelu':
        return nn.LeakyReLU(**kwargs)
    elif act == 'swish':
        return nn.SiLU(**kwargs)
    elif act == 'hswish':
        return nn.Hardswish(**kwargs)
    elif act == 'mish':
        return nn.Mish(**kwargs)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'softmax':
        return nn.Softmax(**kwargs)
    else:
        raise ValueError(f'{act} is not implemented.')

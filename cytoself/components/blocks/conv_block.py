from typing import Optional, Union
from warnings import warn

from torch import nn

from cytoself.components.utils.activation_selecter import act_layer


def calc_groups(in_channels: int, out_channels: int, verbose: bool = True):
    """
    Calculates proper number of groups for Conv2d.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    verbose : bool
        Print user warnings if True

    Returns
    -------
    int
        Number of groups
    """
    if int(in_channels / out_channels) * out_channels == in_channels:
        return out_channels
    else:
        if verbose:
            warn(
                f'in_channels {in_channels} is indivisible by output channel {out_channels}.\n conv_gp is set to 1.',
                UserWarning,
            )
        return 1


class Conv2dBN(nn.Module):
    """
    A set of layers of Conv2d, BachNorm2d and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: Optional[str] = 'swish',
        pad: Union[int, str, tuple] = 'same',
        conv_gp: Union[int, str] = 1,
        dilation: int = 1,
        use_bias: bool = False,
        bn_affine: bool = False,
        name: str = 'conv2dbn',
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the convolving kernel
        stride : int
            Stride of the convolution. Default: 1
        act : str or None
            Name of an activation function or no activation if None
        pad : int or str or tuple
            Padding added to all four sides of the input. Default: 'same'
        conv_gp : int or str
            Number of blocked connections from input channels to output channels.
            Or 'depthwise' for depthwise convolution. Default: 1
        dilation : int
            Spacing between kernel elements. Default: 1
        use_bias : bool
            If True, adds a learnable bias to the output. Default: True
        bn_affine : bool
            If True, batch normalization has learnable affine parameters. Default: False
        name : str
            Name of this block module. Default: conv2dbn
        """
        super().__init__()
        if conv_gp == 'depthwise':
            conv_gp = calc_groups(in_channels, out_channels)

        self.name = name
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=conv_gp,
            bias=use_bias,
        )
        self.bn = None if act is None else nn.BatchNorm2d(out_channels, affine=bn_affine)
        self.act = act if act is None else act_layer(act)

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.bn(x)
            x = self.act(x)
        return x

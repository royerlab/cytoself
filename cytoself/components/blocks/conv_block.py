from torch import nn
from warnings import warn


def calc_groups(in_channels, out_channels):
    if int(out_channels / in_channels) * in_channels == out_channels:
        return int(out_channels / in_channels)
    else:
        warn(
            f'out_channels {out_channels} is indivisible by input channel {in_channels}. '
            f'conv_gp is set to 1.', UserWarning
        )
        return 1


class Conv2dBN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: str = 'swish',
        pad='same',
        conv_gp=1,
        dilation=1,
        use_bias: bool = False,
        bn_affine: bool = False,
        name: str = 'conv2dbn',
    ) -> None:
        """
        2D Convolution block with batch normalization and activation
        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param act: Activation function
        :param pad: Padding added to all four sides of the input. Default: 'same'
        :param conv_gp: Number of blocked connections from input channels to output channels.
        Or 'depthwise' for depthwise convolution. Default: 1
        :param dilation: Spacing between kernel elements. Default: 1
        :param use_bias: If True, adds a learnable bias to the output. Default: True
        :param bn_affine: If True, batch normalization has learnable affine parameters. Default: False
        :param name: Name of this block module. Default: conv2dbn
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
        self.bn = nn.BatchNorm2d(out_channels, affine=bn_affine)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.3)
        elif act == 'swish':
            self.act = nn.SiLU()
        elif act == 'hswish':
            self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

#
#
# class autoencoder(nn.Module):
#     def __init__(self):
#         self.enc = Encoder()
#         self.dec = Decoder()
#
#     def forward(self,x,mode):
#         pass
#
# model = autoencoder()
#
# model.encoder.block1.conv2d
#
# def makemodel(start_index, end_index):
#     short_encoder = nn.Sequential(
#         [model.encoder.block2,
#          model.encoder.block3,
#          model.encoder.block4]
#     )
#
#     short_decoder = nn.Sequential()

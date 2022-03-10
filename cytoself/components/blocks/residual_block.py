from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
from torch import nn
from torch import Tensor
from cytoself.components.blocks.conv_block import Conv2dBN


class ResidualBlockUnit2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str = "swish",
        use_depthwise: bool = False,
        name: str = 'res',
        **kwargs,
    ) -> None:
        """
        2D Residual Block
        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param act: Activation function
        :param use_depthwise: Use depthwise convolution if True.
        :param name: Name of this block module. Default: res
        """
        super().__init__()
        self.name = name
        self.conv1 = Conv2dBN(
            in_channels, out_channels, act=act, pad=kwargs['pad'],
            conv_gp='depthwise' if use_depthwise else 1, name=f'{name}_cvbn1'
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=kwargs['pad'],
            dilation=1,
            groups='depthwise' if use_depthwise else 1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.3)
        elif act == 'swish':
            self.act = nn.SiLU()
        elif act == 'hswish':
            self.act = nn.Hardswish()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.act(x)
        return x


class ResidualBlockRepeat(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_resblocks: int,
            act="swish",
            use_depthwise=False,
            block=None,
            name='res',
            **kwargs,
    ) -> None:
        """
        A series of Residual Blocks
        :param in_channels:
        :param out_channels:
        :param num_resblocks:
        :param act: Activation function
        :param use_depthwise: Use depthwise convolution if True
        :param block: Unit block to repeat
        :param name: Name of this block module. Default: res_repeat
        """
        super().__init__()
        if block is None:
            block = ResidualBlockUnit2d
        self.name = name
        layer_dict = OrderedDict()
        for i in range(num_resblocks):
            layer_dict[f'res{i + 1}'] = block(in_channels, out_channels, act, use_depthwise, **kwargs)
        self.res_repeat = nn.Sequential(layer_dict)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res_repeat(x)
        return x


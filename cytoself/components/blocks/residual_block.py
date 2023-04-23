from collections import OrderedDict
from typing import Optional, Type

from torch import Tensor, nn

from cytoself.components.blocks.conv_block import Conv2dBN
from cytoself.components.utils.activation_selecter import act_layer


class ResidualBlockUnit2d(nn.Module):
    """
    A unit block of 2D residual network.
    """

    def __init__(
        self,
        num_channels: int,
        act: str = "swish",
        use_depthwise: bool = False,
        bn_affine: bool = False,
        name: str = 'res',
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        num_channels : int
            Number of input and output channels. i.e. the resulting module will have the same input & output channels.
        act : str
            Activation function
        use_depthwise : bool
            Use depthwise convolution if True.
        bn_affine : bool
            If True, batch normalization has learnable affine parameters. Default: False
        name : str
            Name of this block module. Default: res
        """
        super().__init__()
        act = act.lower()
        self.name = name
        self.conv1 = Conv2dBN(
            num_channels,
            num_channels,
            act=act,
            conv_gp='depthwise' if use_depthwise else 1,
            bn_affine=bn_affine,
            name=f'{name}_cvbn1',
            **kwargs,
        )
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=num_channels if use_depthwise else 1,
            bias=False,
            padding='same',
            **kwargs,
        )
        self.bn2 = nn.BatchNorm2d(num_channels, affine=bn_affine)
        self.act2 = act_layer(act)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.act2(x)
        return x


class ResidualBlockRepeat(nn.Module):
    """
    A block of repeating residual blocks
    """

    def __init__(
        self,
        num_channels: int,
        num_resblocks: int,
        act: str = "swish",
        use_depthwise: bool = False,
        block: Optional[Type[ResidualBlockUnit2d]] = None,
        name: str = 'res_rpeat',
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        num_channels : int
            Number of input and output channels. i.e. the resulting module will have the same input & output channels.
        num_resblocks : int
            Number of residual blocks
        act : str
            Activation function
        use_depthwise : bool
            Use depthwise convolution if True.
        block : None or resnet block module
            Unit block to repeat
        name : str
            Name of this block module. Default: res
        """
        super().__init__()
        act = act.lower()
        if block is None:
            block = ResidualBlockUnit2d
        self.name = name
        layer_dict = OrderedDict()
        for i in range(num_resblocks):
            layer_dict[f'res{i + 1}'] = block(num_channels, act, use_depthwise, name=f'res{i + 1}', **kwargs)
        self.res_repeat = nn.Sequential(layer_dict)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res_repeat(x)
        return x

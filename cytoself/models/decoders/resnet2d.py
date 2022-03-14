from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
from torch import nn
from torch import Tensor
from cytoself.components.blocks.conv_block import Conv2dBN
from cytoself.components.blocks.residual_block import ResidualBlockRepeat


class DecoderResnet(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        num_residual_layers=2,
        num_hiddens=None,
        num_hidden_decrease=True,
        min_hiddens=1,
        act="swish",
        sampling_mode='bilinear',
        num_blocks=None,
        use_upsampling=True,
        use_depthwise=False,
        name=None,
        **kwargs,
    ) -> None:
        """
        Resnet Decoder
        :param input_shape: Input shape
        :param output_shape: Output shape
        :param num_residual_layers: Number of residual repeat in each residual block
        :param num_hiddens: Number of hidden embeddings (i.e. the channel size right after input shape.)
        :param num_hidden_decrease: Number of channels (i.e. embeddings) will be half at the end of each residual
        block if True.
        :param min_hiddens: Minimum number of hidden embeddings
        :param act: Activation function
        :param sampling_mode: Sampling mode for upsampling
        :param num_blocks: Number of residual blocks
        :param use_upsampling: Upsampling will be used if True
        :param use_depthwise: Depthwise convolution will be used if True
        :param name: Name of this module
        """
        super().__init__()
        input_shape = np.array(input_shape)
        output_shape = np.array(output_shape)
        self.name = name

        # Automatically determine the number of residual blocks
        if num_blocks is None:
            num_blocks = max(np.ceil(np.log2(output_shape[1:] / input_shape[1:])).astype(int))

        self.decoder = nn.ModuleList()
        if num_hiddens is None:
            num_hiddens = input_shape[0]
        else:
            self.decoder.append(Conv2dBN(input_shape[0], num_hiddens, act=act, conv_gp=1, name=f'dec_first', **kwargs))
        _num_hiddens = num_hiddens

        for i in range(num_blocks):
            if use_upsampling:
                target_shape = tuple(np.ceil(output_shape[1:] / (2 ** (num_blocks - (i + 1)))).astype(int))
                self.decoder.append(nn.Upsample(size=target_shape, mode=sampling_mode, align_corners=False))

            self.decoder.append(
                ResidualBlockRepeat(
                    num_hiddens, num_residual_layers, act=act, use_depthwise=use_depthwise,
                    name=f'res{i+1}', **kwargs,
                )
            )

            if num_hidden_decrease:
                _num_hiddens = max(int(num_hiddens / 2), min_hiddens)
            if i == num_blocks - 1:
                _num_hiddens = output_shape[0]
            self.decoder.append(
                Conv2dBN(num_hiddens, _num_hiddens, act=act, conv_gp=1, name=f'res{i + 1}last', **kwargs)
            )
            num_hiddens = _num_hiddens

    def forward(self, x: Tensor) -> Tensor:
        for lyr in self.decoder:
            x = lyr(x)
        return x

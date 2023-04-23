from typing import Optional

import numpy as np
from torch import Tensor, nn

from cytoself.components.blocks.conv_block import Conv2dBN
from cytoself.components.blocks.residual_block import ResidualBlockRepeat


class DecoderResnet(nn.Module):
    """
    Resnet model as a decoder.
    """

    def __init__(
        self,
        input_shape: tuple,
        output_shape: tuple,
        num_residual_layers: int = 2,
        num_hiddens: Optional[int] = None,
        num_hidden_decrease: bool = True,
        min_hiddens: int = 1,
        act: str = "swish",
        sampling_mode: str = 'bilinear',
        num_blocks: Optional[int] = None,
        use_upsampling: bool = True,
        use_depthwise: bool = False,
        name: str = 'decoder',
        linear_output: bool = True,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        input_shape : tuple
            Input tensor shape: (Channel, Height, Width)
        output_shape : tuple
            Output tensor shape: (Channel, Height, Width)
        num_residual_layers : int
            Number of residual repeat in each residual block. Default: 2
        num_hiddens : int or None
            Number of hidden embeddings (i.e. the channel size right after input shape.)
            If None, it will be same as the input channel size. Default: None
        num_hidden_decrease : bool
            Number of channels (i.e. embeddings) will be halved at the end of each residual block if True until it
            reaches the min_hiddens. Default: True.
        min_hiddens : int
            Minimum number of hidden embeddings. Default: 1
        act : str
            Activation function
        sampling_mode : str
            Sampling mode for upsampling. Default: bilinear
        num_blocks : int or None
            Number of the blocks of repeating residual blocks
        use_upsampling : bool
            An upsampling layer will be added before each repeating residual block if True
        use_depthwise : bool
            Use depthwise convolution if True.
        name : str
            Name of this block module. Default: res
        linear_output : bool
            No activation function is used at the last layer if True, otherwise the same activation is used.
        """
        super().__init__()
        input_shape = np.array(input_shape)
        output_shape = np.array(output_shape)
        act = act.lower()
        self.name = name

        # Automatically determine the number of residual blocks
        if num_blocks is None:
            num_blocks = max(np.ceil(np.log2(output_shape[1:] / input_shape[1:])).astype(int))

        self.decoder = nn.ModuleDict()
        if num_hiddens is None:
            num_hiddens = input_shape[0]
        else:
            self.decoder['dec_first'] = Conv2dBN(
                input_shape[0],
                num_hiddens,
                act=act,
                conv_gp=1,
                name='dec_first',
                **kwargs,
            )
        _num_hiddens = num_hiddens

        for i in range(num_blocks):
            if use_upsampling:
                target_shape = tuple(np.ceil(output_shape[1:] / (2 ** (num_blocks - (i + 1)))).astype(int))
                self.decoder[f'up{i + 1}'] = nn.Upsample(size=target_shape, mode=sampling_mode, align_corners=False)

            self.decoder[f'resrep{i+1}'] = ResidualBlockRepeat(
                num_hiddens,
                num_residual_layers,
                act=act,
                use_depthwise=use_depthwise,
                name=f'res{i+1}',
                **kwargs,
            )

            if num_hidden_decrease:
                _num_hiddens = max(int(num_hiddens / 2), min_hiddens)
            self.decoder[f'resrep{i+1}last'] = Conv2dBN(
                num_hiddens,
                output_shape[0] if (i == num_blocks - 1 and not linear_output) else _num_hiddens,
                act=act,
                conv_gp=1,
                name=f'resrep{i + 1}last',
                **kwargs,
            )
            num_hiddens = _num_hiddens

            if i == num_blocks - 1 and linear_output:
                self.decoder['output_conv'] = nn.Conv2d(
                    _num_hiddens,
                    output_shape[0],
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    dilation=1,
                    groups=1,
                    bias=False,
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        for _, lyr in self.decoder.items():
            x = lyr(x)
        return x

import copy
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

from torch import Tensor, nn
from torchvision.models._utils import _make_divisible
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.utils import _log_api_usage_once

"""
This is adapted from torchvision.models.efficientnet
"""

__all__ = [
    "EfficientNet",
    "efficientenc_b0",
    "efficientenc_b1",
    "efficientenc_b2",
    "efficientenc_b3",
    "efficientenc_b4",
    "efficientenc_b5",
    "efficientenc_b6",
    "efficientenc_b7",
]


default_block_args = [
    {
        'expand_ratio': 1,
        'kernel': 3,
        'stride': 1,
        'input_channels': 32,
        'out_channels': 16,
        'num_layers': 1,
    },
    {
        'expand_ratio': 6,
        'kernel': 3,
        'stride': 2,
        'input_channels': 16,
        'out_channels': 24,
        'num_layers': 2,
    },
    {
        'expand_ratio': 6,
        'kernel': 5,
        'stride': 2,
        'input_channels': 24,
        'out_channels': 40,
        'num_layers': 2,
    },
    {
        'expand_ratio': 6,
        'kernel': 3,
        'stride': 2,
        'input_channels': 40,
        'out_channels': 80,
        'num_layers': 3,
    },
    {
        'expand_ratio': 6,
        'kernel': 5,
        'stride': 1,
        'input_channels': 80,
        'out_channels': 112,
        'num_layers': 3,
    },
    {
        'expand_ratio': 6,
        'kernel': 5,
        'stride': 2,
        'input_channels': 112,
        'out_channels': 192,
        'num_layers': 4,
    },
    {
        'expand_ratio': 6,
        'kernel': 3,
        'stride': 1,
        'input_channels': 192,
        'out_channels': 320,
        'num_layers': 1,
    },
]


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        stochastic_depth_prob: float = 0.2,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        first_layer_stride: int = 2,
    ) -> None:
        """
        EfficientNet main class

        Parameters
        ----------
        inverted_residual_setting : List[MBConvConfig]
            A list of MBConvConfig which consists of the EfficientNet model.
        in_channels : int
            Input channel size for the entire model.
            If given, an additional ConvBN layer will be added in the beginning.
        out_channels : int
            Output channel size for the entire model. If given, an additional ConvBN layer will be added in the end.
        stochastic_depth_prob : float
            The stochastic depth probability
        block
            Module specifying inverted residual building block for mobilenet
        norm_layer
            Module specifying the normalization layer to use
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building the first layer
        if in_channels is not None:
            firstconv_output_channels = inverted_residual_setting[0].input_channels
            layers.append(
                ConvNormActivation(
                    in_channels,
                    firstconv_output_channels,
                    kernel_size=3,
                    stride=first_layer_stride,
                    norm_layer=norm_layer,
                    activation_layer=nn.SiLU,
                )
            )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building the last layer
        if out_channels is not None:
            layers.append(
                ConvNormActivation(
                    inverted_residual_setting[-1].out_channels,
                    out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.SiLU,
                )
            )

        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


def _efficientnet(blocks_args: List[dict], width_mult: float, depth_mult: float, **kwargs: Any) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = []
    for arg in blocks_args:
        inverted_residual_setting.append(bneck_conf(**arg))
    model = EfficientNet(inverted_residual_setting, **kwargs)
    return model


def efficientenc_b0(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B0 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(blocks_args, 1.0, 1.0, **kwargs)


def efficientenc_b1(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B1 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(blocks_args, 1.0, 1.1, **kwargs)


def efficientenc_b2(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B2 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(blocks_args, 1.1, 1.2, **kwargs)


def efficientenc_b3(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B3 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(blocks_args, 1.2, 1.4, **kwargs)


def efficientenc_b4(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B4 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(blocks_args, 1.4, 1.8, **kwargs)


def efficientenc_b5(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B5 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(
        blocks_args,
        1.6,
        2.2,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientenc_b6(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B6 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(
        blocks_args,
        1.8,
        2.6,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientenc_b7(blocks_args: Optional[List[dict]] = None, **kwargs: Any) -> EfficientNet:
    """
    Constructs an encoder with EfficientNet B7 architecture using given block arguments.

    Parameters
    ----------
    blocks_args : list of dict
        A list of dict of block arguments. See default_block_args for what arguments are needed.

    Returns
    -------
    An EfficientNet model

    """
    if blocks_args is None:
        blocks_args = default_block_args
    return _efficientnet(
        blocks_args,
        2.0,
        3.1,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )

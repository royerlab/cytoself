import numpy as np

from tensorflow.compat.v1.keras.layers import (
    Input,
    Conv2D,
    Conv3D,
    Conv2DTranspose,
    BatchNormalization,
    Add,
    Activation,
    LeakyReLU,
    PReLU,
    UpSampling2D,
    UpSampling3D,
    ZeroPadding2D,
    ZeroPadding3D,
    DepthwiseConv2D,
)
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K

from cytoself.components.blocks.conv_block import conv2d_bn, Swish


def residual_stack2D(
    h,  # input tensor
    num_hiddens,  # number of output filters
    num_residual_layers=3,  # number of residual layers
    act="lrelu",
    use_depthwise=False,
    name="res",
):
    for i in range(num_residual_layers):
        h_i = conv2d_bn(
            h,
            num_hiddens,
            kernel_size=3,
            stride=1,
            act=act,
            use_depthwise=use_depthwise,
            name=f"{name}3x3_{i:d}",
        )
        depth_multiplier = int(num_hiddens / max(K.int_shape(h_i)[-1], 1)) > 0
        use_depthwise = depth_multiplier > 0 and use_depthwise
        if use_depthwise:
            h_i = DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
                depth_multiplier=depth_multiplier,
                name=f"{name}1x1_{i:d}_cv",
            )(h_i)
        else:
            h_i = Conv2D(
                filters=num_hiddens,
                kernel_size=3,
                strides=1,
                padding="same",
                name=f"{name}1x1_{i:d}_cv",
            )(h_i)
        h_i = BatchNormalization(name=f"{name}1x1_{i:d}_bn")(h_i)
        h = Add(name=f"{name}add_{i:d}")([h, h_i])
        if act == "lrelu":
            h = LeakyReLU(name=f"{name}act_{i:d}")(h)
        elif act == "swish":
            h = Swish(name=f"{name}act_{i:d}")(h)
        else:
            h = Activation(act, name=f"{name}act_{i:d}")(h)
    return h


def Decoderres_bicub(
    input_dim,
    output_dim,
    num_hiddens=1024,
    num_residual_layers=3,
    num_hidden_decrease=True,
    min_hiddens=1,
    act="lrelu",
    include_first=True,
    include_last=True,
    name=None,
    num_blocks=None,
    use_upsampling=True,
    use_depthwise=False,
):
    x = Input(input_dim, name="dec_in")

    input_dim = np.array(input_dim)
    output_dim = np.array(output_dim)
    if num_blocks is None:
        num_blocks = max(np.ceil(np.log2(output_dim[:-1] / input_dim[:-1])).astype(int))

    if include_first:
        h = conv2d_bn(
            x,
            num_hiddens,
            kernel_size=3,
            stride=1,
            act=act,
            use_depthwise=use_depthwise,
            name="dec_cv1",
        )
    else:
        h = x
    for i in range(num_blocks):
        if use_upsampling:
            h = UpSampling2D(interpolation="bilinear")(h)
        h = residual_stack2D(
            h,
            num_hiddens,
            num_residual_layers,
            act=act,
            use_depthwise=use_depthwise,
            name=f"res{i}",
        )
        if num_hidden_decrease:
            num_hiddens = max(int(num_hiddens / 2), min_hiddens)
        if not include_last and i == num_blocks - 1:
            num_hiddens = output_dim[-1]

        # Determine padding
        if use_upsampling:
            current_dim = K.int_shape(h)[1:-1]
            target_dim = np.ceil(output_dim[:-1] / (2 ** (num_blocks - (i + 1))))
            diff = (current_dim - target_dim).astype(
                int
            )  # difference b/w current and target dim
            pad = abs(2 - diff)  # convert diff to padding
            padding = [
                (np.ceil(i / 2).astype(int), np.floor(i / 2).astype(int)) for i in pad
            ]

            h = ZeroPadding2D(padding=padding, name=f"pad{i}")(h)
        h = conv2d_bn(
            h,
            num_hiddens,
            kernel_size=3,
            stride=1,
            act=act,
            use_depthwise=use_depthwise,
            name=f"res{i}last",
            pad="valid" if use_upsampling else "same",
        )
    if include_last:
        depth_multiplier = int(output_dim[-1] / max(K.int_shape(h)[-1], 1))
        use_depthwise = depth_multiplier > 0 and use_depthwise
        if use_depthwise:
            h = DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
                depth_multiplier=depth_multiplier,
                name="dec_cvt_last",
            )(h)
        else:
            h = Conv2D(
                filters=output_dim[-1],
                kernel_size=3,
                strides=1,
                padding="same",
                name="dec_cvt_last",
            )(h)
    return Model(x, h, name=name)

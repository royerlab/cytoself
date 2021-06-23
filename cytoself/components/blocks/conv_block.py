from tensorflow.compat.v1.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    BatchNormalization,
    Activation,
    LeakyReLU,
    Lambda,
    Reshape,
    multiply,
    DepthwiseConv2D,
)
import tensorflow.compat.v1.keras.backend as K

from cytoself.components.layers.activation import Swish


def conv2d_bn(
    inputs,
    num_outputs,
    kernel_size=3,
    stride=1,
    act="relu",
    pad="same",
    use_depthwise=False,
    name=None,
):
    depth_multiplier = int(num_outputs / max(K.int_shape(inputs)[-1], 1))
    use_depthwise = depth_multiplier > 0 and use_depthwise
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_cv"
        act_name = name + "_" + act
    else:
        bn_name = None
        conv_name = None
        act_name = None
    if use_depthwise:
        conv = DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            padding=pad,
            depth_multiplier=depth_multiplier,
            name=conv_name,
        )(inputs)
    else:
        conv = Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=pad,
            name=conv_name,
        )(inputs)
    conv = BatchNormalization(name=bn_name)(conv)
    if act == "relu":
        conv = Activation(act, name=act_name)(conv)
    elif act == "lrelu":
        conv = LeakyReLU(alpha=0.3, name=act_name)(conv)
    elif act == "swish":
        conv = Swish(name=act_name)(conv)
    return conv

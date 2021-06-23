"""
This code is adapted from https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py.
"""
import os
import math
from tensorflow.compat.v1.nn import swish
from tensorflow.compat.v1.keras import backend, layers, models, utils


encoder_block_args = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

decoder_block_args = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 256,
        "filters_out": 128,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 128,
        "filters_out": 64,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 64,
        "filters_out": 32,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 32,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
]


CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        "distribution": "untruncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


def block_dw(
    inputs,
    activation_fn=swish,
    dropout_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """A mobile inverted residual block.

    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        dropout_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "expand_conv",
        )(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + "expand_bn")(x)
        x = layers.Activation(activation_fn, name=name + "expand_activation")(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(backend, x, kernel_size), name=name + "dwconv_pad"
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "dwconv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "bn")(x)
    x = layers.Activation(activation_fn, name=name + "activation")(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation_fn,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_expand",
        )(se)
        x = layers.multiply([x, se], name=name + "se_excite")

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "project_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "project_bn")(x)
    if id_skip is True and strides == 1 and filters_in == filters_out:
        if dropout_rate > 0:
            x = layers.Dropout(
                dropout_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs], name=name + "add")

    return x


def block_up(
    inputs,
    activation_fn=swish,
    dropout_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """A mobile inverted residual block.

    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        dropout_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "expand_conv",
        )(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + "expand_bn")(x)
        x = layers.Activation(activation_fn, name=name + "expand_activation")(x)
    else:
        x = inputs
    conv_pad = "same"
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "upconv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "bn")(x)
    x = layers.Activation(activation_fn, name=name + "activation")(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation_fn,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_expand",
        )(se)
        x = layers.multiply([x, se], name=name + "se_excite")

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "project_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "project_bn")(x)
    if id_skip is True and strides == 1 and filters_in == filters_out:
        if dropout_rate > 0:
            x = layers.Dropout(
                dropout_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs], name=name + "add")

    return x


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    dropout_rate=0.0,  # 0.2
    drop_connect_rate=0.0,  # 0.2
    depth_divisor=8,
    activation_fn=swish,
    blocks_args=encoder_block_args,
    model_name="efficientnet",
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    isencoder=1,  # 0: decoder, 1: normal encoder, 2: encoder w/o entry conv2d, 3: encoder w/o exit conv2d
    num_hiddens_last=1280,
    **kwargs
):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # global backend, layers, models, keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # Can be removed in the future(?)
    if not (weights is None or os.path.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), "
            "or the path to the weights file to be loaded."
        )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build entry
    x = img_input
    if isencoder == 1 or isencoder == 3:
        x = layers.ZeroPadding2D(
            padding=correct_pad(backend, x, 3), name="stem_conv_pad"
        )(x)
        x = layers.Conv2D(
            round_filters(32),
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
        x = layers.Activation(activation_fn, name="stem_activation")(x)
    elif isencoder == 2:
        pass
    else:
        if include_top:
            x = layers.Reshape((1, 1, input_shape[-1]), name="top_reshape")(x)
            x = layers.UpSampling2D(
                (6, 6), interpolation="bilinear", name="top_upsmpl"
            )(x)
        x = layers.Conv2D(
            round_filters(512),
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
        x = layers.Activation(activation_fn, name="top_activation")(x)

    # Build blocks
    from copy import deepcopy

    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args["repeats"] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args["repeats"] > 0
        # Update block input and output filters based on depth multiplier.
        args["filters_in"] = round_filters(args["filters_in"])
        args["filters_out"] = round_filters(args["filters_out"])

        for j in range(round_repeats(args.pop("repeats"))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            if isencoder:
                x = block_dw(
                    x,
                    activation_fn,
                    drop_connect_rate * b / blocks,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                    **args
                )
            else:
                x = block_up(
                    x,
                    activation_fn,
                    drop_connect_rate * b / blocks,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                    **args
                )
            b += 1

    # Build exit
    if isencoder == 1 or isencoder == 2:
        x = layers.Conv2D(
            round_filters(num_hiddens_last),
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
        x = layers.Activation(activation_fn, name="top_activation")(x)
        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)
    elif isencoder == 3:
        pass
    else:
        x = layers.Conv2DTranspose(
            round_filters(32),
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
        x = layers.Activation(activation_fn, name="stem_activation")(x)
        x = layers.Conv2D(
            1,
            kernel_size=3,
            padding="valid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="dec_last",
        )(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def EfficientEncoderB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.0,
    depth_coefficient=1.0,
    dropout_rate=0.2,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b0",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientEncoderB1(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.0,
    depth_coefficient=1.1,
    dropout_rate=0.2,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b5",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientEncoderB2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.1,
    depth_coefficient=1.2,
    dropout_rate=0.3,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b5",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientEncoderB3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.2,
    depth_coefficient=1.4,
    dropout_rate=0.3,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b5",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientEncoderB4(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.4,
    depth_coefficient=1.8,
    dropout_rate=0.4,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b5",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientEncoderB5(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.6,
    depth_coefficient=2.2,
    dropout_rate=0.4,
    blocks_args=encoder_block_args,
    isencoder=1,
    num_hiddens_last=1280,
    name="efficientencoder-b5",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        num_hiddens_last=num_hiddens_last,
        **kwargs
    )


def EfficientDecoderB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    width_coefficient=1.0,
    depth_coefficient=1.0,
    dropout_rate=0,
    blocks_args=decoder_block_args,
    isencoder=0,
    name="efficientdecoder-b0",
    **kwargs
):
    return EfficientNet(
        width_coefficient,
        depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=name,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        blocks_args=blocks_args,
        isencoder=isencoder,
        **kwargs
    )

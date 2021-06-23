from collections.abc import Iterable
from tensorflow.compat.v1.keras.layers import (
    Input,
    Dense,
    Flatten,
    Concatenate,
    Dropout,
)
from tensorflow.compat.v1.keras.models import Model


def fc_block(
    in_shape,
    num_layers,
    n_units,
    n_clss,
    activation="relu",
    last_act="softmax",
    dropout_rate=0,
    name="fcblock",
):
    """
    An fc block.
    :param in_shape: input shape to the fc block
    :param num_layers: number of layers in the fc block
    :param n_units: number of units in each fc layer
    :param n_clss: number of classes at the output layer
    :param activation: activation function
    :param last_act: activation function at the last layer
    :param dropout_rate: dropout rate
    :param name: name of the fc block
    :return: a model of fc layers
    """
    if isinstance(in_shape[0], Iterable):
        x = [Input(sh, name=f"fc_in{i + 1}") for i, sh in enumerate(in_shape)]
        y = []
        for i, x0 in enumerate(x):
            if len(in_shape[i]) > 2:
                y0 = Flatten(name=f"flat{i + 1}")(x0)
            else:
                y0 = x0
            y.append(y0)
        y = Concatenate(name=f"fc_cat")(y)
    else:
        x = Input(in_shape, name="fc_in")
        if len(in_shape) > 2:
            y = Flatten(name=f"flat")(x)
        else:
            y = x

    for i in range(num_layers):
        y = Dense(
            n_units if i < num_layers - 1 else n_clss,
            activation if i < num_layers - 1 else last_act,
            name=f"fc_{i}",
        )(y)
        if i < num_layers - 1 and dropout_rate > 0:
            y = Dropout(dropout_rate, name=f"drop{i}")(y)
    return Model(x, y, name=name)

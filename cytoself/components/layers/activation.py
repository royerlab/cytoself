from tensorflow.compat.v1.nn import swish
from tensorflow.compat.v1.keras.layers import Lambda


def Swish(name=None):
    return Lambda(swish, name=name)

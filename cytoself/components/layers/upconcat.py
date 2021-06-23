import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.keras.layers import (
    Input,
    Add,
    UpSampling2D,
    Concatenate,
    Layer,
)


class UpConcat2D(Layer):
    def __init__(self, outdim, mergetype="cat", interpolation="bilinear", **kwargs):
        """
        Upsample tensors upto outdim and merge them.
        :param outdim: output 2D shape
        :param tensors: a list of tensors
        :param mergetype: Concatenate or Add
        :param interpolation: mode of interpolation; nearest or bilinear
        :param name: layer name
        :return: a merged tensor
        """
        super().__init__(**kwargs)
        self.outdim = outdim
        self.mergetype = mergetype
        self.interpolation = interpolation

    def build(self, input_shape):
        super(UpConcat2D, self).build(input_shape)

    def call(self, tensors):
        tsrup_list = []
        for ii, tsr in enumerate(tensors):
            tsrup = tfv1.image.resize(tsr, self.outdim, method=self.interpolation)
            tsrup_list.append(tsrup)
        if self.mergetype == "cat":
            tsrout = Concatenate()(tsrup_list)
        elif self.mergetype == "add":
            tsrout = Add()(tsrup_list)
        else:
            tsrout = tsrup_list
        return tsrout

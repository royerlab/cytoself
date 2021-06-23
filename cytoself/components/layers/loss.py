import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.keras.layers import Layer
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.losses import MSE
from tensorflow.compat.v1.math import squared_difference, logical_not
from tensorflow.python.keras.metrics import MeanMetricWrapper


class SimpleMSE(Layer):
    def __init__(self, coeff=1, **kwargs):
        self.coeff = coeff
        self.mse_loss = None
        super(SimpleMSE, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[0][1:] == input_shape[1][1:]
        super(SimpleMSE, self).build(input_shape)

    def call(self, inputs):
        x1, x2 = inputs
        self.mse_loss = K.mean(squared_difference(x1, x2)) * self.coeff
        self.add_loss(self.mse_loss)
        return x1

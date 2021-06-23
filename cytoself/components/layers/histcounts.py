import math
import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.keras.layers import Layer


class Histcounts(Layer):
    def __init__(self, value_range, nbins=100, **kwargs):  # n_parallel=10,
        """
        Histogram counting layer
        :param value_range: range of edge [low, hight]
        :param nbins: number of bins
        :param kwargs: kwargs for a layer object
        """
        super().__init__(**kwargs)
        self.value_range = [math.floor(value_range[0]), math.ceil(value_range[1])]
        self.nbins = nbins
        # self.n_parallel = n_parallel

    def build(self, input_sahpe):
        super(Histcounts, self).build(input_sahpe)

    def call(self, x):
        # hcunt = tfv1.vectorized_map(  # ValueError: No converter defined for HistogramFixedWidth
        hcunt = tfv1.map_fn(
            lambda x: tfv1.histogram_fixed_width(x, self.value_range, self.nbins),
            tfv1.cast(x, tfv1.int32),
        )
        return tfv1.cast(hcunt, tfv1.float32)

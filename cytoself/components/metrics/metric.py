from tensorflow.python.keras.metrics import MeanMetricWrapper


# metric module to monitor arbitrary loss
class Metric(MeanMetricWrapper):
    """
    A metric module to monitor arbitrary loss.
    """

    def __init__(self, metric, name, dtype=None, **kwargs):
        self.metric = metric
        # self.name = name
        # super(Metric, self).__init__(self.metric_func(), name, dtype=dtype, **kwargs)
        super(Metric, self).__init__(lambda x, y: metric, name, dtype=dtype, **kwargs)

    # there seems to be a bug in the tf.keras implementation
    def get_config(self):
        return {"metric": self.metric_func(), "name": self.name, "dtype": self.dtype}

    # Can be removed
    def metric_func(self):
        def pass_metric(y_true, y_pred):
            return self.metric

        return pass_metric

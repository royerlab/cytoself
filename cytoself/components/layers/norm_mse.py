from tensorflow.compat.v1.keras.losses import MSE


def normalized_mse(var):
    def loss(y_true, y_pred):
        return MSE(y_true, y_pred) / var

    return loss

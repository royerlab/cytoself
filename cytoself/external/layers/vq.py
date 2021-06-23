"""
This code is adapted from https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
"""
import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.train import ExponentialMovingAverage
from tensorflow.compat.v1.keras.layers import Layer
from tensorflow.compat.v1.keras import initializers
import tensorflow.compat.v1.keras.backend as K


class VectorQuantizer(Layer):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        n_outputs=1,
        coeff=1,
        **kwargs
    ):
        """
        Vector Quantize layer
        :param embedding_dim: dimension size for each embedding
        :param num_embeddings: total number of quantized embedding
        :param commitment_cost: commitment cost
        :param n_outputs: output quantized vector or index or onehot vector
        :param coeff: coefficient for VQ loss; used for adjusting the balance between other losses in a model
        :param kwargs: kwargs for Layer object
        """
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self.encodings = None
        self.encoding_indices = None
        self.perplexity = None
        self.e_latent_loss = None  # commitment loss
        self.q_latent_loss = None  # quantization loss
        self.n_outputs = n_outputs  # output both quantized & encoding_indices
        self.coeff = coeff  # coefficient applied to all loss
        self._w = 0
        self.in_shape = None
        super(VectorQuantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.in_shape = input_shape
        assert input_shape[-1] == self._embedding_dim
        # Add embedding weights.
        self._w = self.add_weight(
            name="embedding_vq",
            shape=(self._embedding_dim, self._num_embeddings),
            initializer="uniform",
            trainable=True,
        )
        # Finalize building.
        super(VectorQuantizer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self._embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (
            K.sum(flat_inputs ** 2, axis=1, keepdims=True)
            - 2 * K.dot(flat_inputs, self._w)
            + K.sum(self._w ** 2, axis=0, keepdims=True)
        )

        # Retrieve encoding indices.
        self.encoding_indices = K.argmax(-distances, axis=1)
        self.encodings = K.one_hot(self.encoding_indices, self._num_embeddings)
        self.encoding_indices = K.reshape(self.encoding_indices, K.shape(x)[:-1])
        encodings_reshape = K.one_hot(self.encoding_indices, self._num_embeddings)
        quantized = self.quantize(self.encoding_indices)

        self.e_latent_loss = K.mean((K.stop_gradient(quantized) - x) ** 2)
        self.q_latent_loss = K.mean((quantized - K.stop_gradient(x)) ** 2)
        loss = (
            self.q_latent_loss + self._commitment_cost * self.e_latent_loss
        ) * self.coeff

        quantized = x + K.stop_gradient(quantized - x)
        x_loss_copy = tfv1.cast(K.mean(x, axis=-1), tfv1.int64)
        self.encoding_indices = x_loss_copy + K.stop_gradient(
            self.encoding_indices - x_loss_copy
        )
        x_loss_copy2 = K.mean(x, axis=-1, keepdims=True)
        encodings_reshape = x_loss_copy2 + K.stop_gradient(
            encodings_reshape - x_loss_copy2
        )
        # Important Note:
        # 	This step is used to copy the gradient from inputs to quantized.

        # Metrics.
        avg_probs = K.mean(self.encodings, axis=0)
        self.perplexity = K.exp(-K.sum(avg_probs * K.log(avg_probs + 1e-10)))

        self.add_loss(loss)
        if self.n_outputs == 1:
            return quantized
        elif self.n_outputs == 2:
            return quantized, self.encoding_indices
        else:
            return quantized, self.encoding_indices, encodings_reshape

        # "quantized" are the quantized outputs of the encoder.
        # That is also what is used during training with the straight-through estimator.
        # To get the one-hot coded assignments use "encodings" instead.
        # These encodings will not pass gradients into to encoder,
        # but can be used to train a PixelCNN on top afterwards.

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tfv1.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

    # This is to support model.save() but currently not working.
    def get_config(self):
        return {
            "w": self._w,
            "embedding_dim": self._embedding_dim,
            "num_embeddings": self._num_embeddings,
            "commitment_cost": self._commitment_cost,
            "encodings": self.encodings,
            "encoding_indices": self.encoding_indices,
            "perplexity": self.perplexity,
            "e_latent_loss": self.e_latent_loss,
            "q_latent_loss": self.q_latent_loss,
            "n_outputs": self.n_outputs,
            "coeff": self.coeff,
            "emb": self.embeddings,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VectorQuantizerEMA(Layer):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        decay,
        epsilon=1e-5,
        **kwargs
    ):
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._decay = decay
        self._commitment_cost = commitment_cost
        self._epsilon = epsilon
        self.encodings = None
        self.encoding_indices = None
        self.perplexity = None
        self.e_latent_loss = None  # commitment loss
        self.q_latent_loss = None  # quantization loss
        super(VectorQuantizerEMA, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[-1] == self._embedding_dim
        initializer = initializers.RandomUniform(seed=0)
        # w is a matrix with an embedding in each column. When training, the
        # embedding is assigned to be the average of all inputs assigned to that
        # embedding.
        self._w = self.add_weight(
            name="embedding_vq",
            shape=(self._embedding_dim, self._num_embeddings),
            initializer=initializer,
            trainable=False,
        )

        self._ema_cluster_size = self.add_weight(
            "ema_cluster_size",
            shape=[self._num_embeddings],
            initializer="zeros",
            trainable=False,
        )
        self._ema_w = self.add_weight(
            "ema_dw",
            # initializer=self._w.initialized_value(),
            initializer=initializer,
            trainable=False,
        )

        # Finalize building.
        super(VectorQuantizerEMA, self).build(input_shape)

    def call(self, x, training=None):
        # Ensure that the weights are read fresh for each timestep, which otherwise
        # would not be guaranteed in an RNN setup. Note that this relies on inputs
        # having a data dependency with the output of the previous timestep - if
        # this is not the case, there is no way to serialize the order of weight
        # updates within the module, so explicit external dependencies must be used.
        with tfv1.control_dependencies([x]):
            w = self._w.read_value()

        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self._embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (
            K.sum(flat_inputs ** 2, axis=1, keepdims=True)
            - 2 * K.dot(flat_inputs, w)
            + K.sum(w ** 2, axis=0, keepdims=True)
        )

        # Retrieve encoding indices.
        self.encoding_indices = K.argmax(-distances, axis=1)
        self.encodings = K.one_hot(self.encoding_indices, self._num_embeddings)
        self.encoding_indices = K.reshape(self.encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(self.encoding_indices)
        self.e_latent_loss = K.mean((K.stop_gradient(quantized) - x) ** 2)

        if training:
            ema = ExponentialMovingAverage(self._decay)

            updated_ema_cluster_size = ema.apply(
                [self._ema_cluster_size, K.mean(self.encodings, 0)]
            )
            dw = K.dot(K.transpose(flat_inputs), self.encodings)
            updated_ema_w = ema.apply([self._ema_w, dw,])
            n = K.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )
            normalised_updated_ema_w = updated_ema_w / K.reshape(
                updated_ema_cluster_size, [1, -1]
            )

            with tfv1.control_dependencies([self.e_latent_loss]):
                update_w = tfv1.assign(self._w, normalised_updated_ema_w)
                with tfv1.control_dependencies([update_w]):
                    loss = self._commitment_cost * self.e_latent_loss

        else:
            loss = self._commitment_cost * self.e_latent_loss
        quantized = x + K.stop_gradient(quantized - x)
        # 	This step is used to copy the gradient from inputs to quantized.

        avg_probs = K.mean(self.encodings, 0)
        self.perplexity = K.exp(-K.sum(avg_probs * K.log(avg_probs + 1e-10)))

        self.add_loss(loss)
        return quantized

        # "quantized" are the quantized outputs of the encoder.
        # That is also what is used during training with the straight-through estimator.
        # To get the one-hot coded assignments use "encodings" instead.
        # These encodings will not pass gradients into to encoder,
        # but can be used to train a PixelCNN on top afterwards.

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tfv1.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

    def compute_output_shape(self, input_shape):
        return input_shape

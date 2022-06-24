from typing import Optional
from torch import Tensor

from cytoself.components.layers.vq import VectorQuantizer
from cytoself.trainer.autoencoder.vanilla import VanillaAE


class VQVAE(VanillaAE):
    """
    Vector Quantized Variational Autoencoder model
    """

    def __init__(
        self,
        emb_shape: tuple[int, int, int],
        vq_args: dict,
        input_shape: Optional[tuple[int, int, int]] = None,
        output_shape: Optional[tuple[int, int, int]] = None,
        encoder_args: Optional[dict] = None,
        decoder_args: Optional[dict] = None,
        encoder: Optional = None,
        decoder: Optional = None,
    ):
        """
        Constructs a VQVAE model

        Parameters
        ----------
        emb_shape : tuple
            Embedding tensor shape
        vq_args : dict
            Additional arguments for the Vector Quantization layer
        input_shape : tuple
            Input tensor shape
        output_shape : tuple
            Output tensor shape
        encoder_args : dict
            Additional arguments for encoder
        decoder_args : dict
            Additional arguments for decoder
        encoder : encoder module
            (Optional) Custom encoder module
        decoder : decoder module
            (Optional) Custom decoder module
        """
        super().__init__(emb_shape, input_shape, output_shape, encoder_args, decoder_args, encoder, decoder)
        self.vq_layer = VectorQuantizer(embedding_dim=emb_shape[0], **vq_args)
        self.vq_loss = None
        self.perplexity = None
        self.encoding_onehot = None
        self.encoding_indices = None
        self.index_histogram = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        (
            self.vq_loss,
            x,
            self.perplexity,
            self.encoding_onehot,
            self.encoding_indices,
            self.index_histogram,
        ) = self.vq_layer(x)
        x = self.decoder(x)
        return x

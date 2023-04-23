from typing import Optional

from torch import Tensor

from cytoself.components.layers.vq import VectorQuantizer
from cytoself.trainer.autoencoder.base import BaseAE
from cytoself.trainer.autoencoder.cytoselffull import calc_emb_dim


class VQVAE(BaseAE):
    """
    Vector Quantized Variational Autoencoder model
    """

    def __init__(
        self,
        emb_shape: tuple[int, int],
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
        vq_args, emb_shape = calc_emb_dim([vq_args], [emb_shape])
        self.vq_args, self.emb_shape = vq_args[0], emb_shape[0]
        super().__init__(self.emb_shape, input_shape, output_shape, encoder_args, decoder_args, encoder, decoder)
        self.mse_loss = None
        self.vq_layer = VectorQuantizer(**self.vq_args)
        self.vq_loss = None
        self.perplexity = None

    def forward(self, x: Tensor, output_layer: str = 'decoder') -> Tensor:
        x = self.encoder(x)
        if output_layer == 'encoder':
            return x
        (
            self.vq_loss,
            quantized,
            self.perplexity,
            _,
            encoding_indices,
            index_histogram,
            _,
        ) = self.vq_layer(x)
        if output_layer == 'vqvec':
            return quantized
        elif output_layer == 'vqind':
            return encoding_indices
        elif output_layer == 'vqindhist':
            return index_histogram

        x = self.decoder(quantized)
        return x

from typing import Optional

import numpy as np
from torch import Tensor

from cytoself.components.blocks.fc_block import FCblock
from cytoself.trainer.autoencoder.vqvae import VQVAE


class VQVAEFC(VQVAE):
    """
    Vector Quantized Variational Autoencoder with an FC block
    """

    def __init__(
        self,
        emb_shape: tuple[int, int],
        vq_args: dict,
        num_class: int,
        input_shape: Optional[tuple[int, int, int]] = None,
        output_shape: Optional[tuple[int, int, int]] = None,
        fc_input_type: str = 'vqvec',
        encoder_args: Optional[dict] = None,
        decoder_args: Optional[dict] = None,
        fc_args: Optional[dict] = None,
        encoder: Optional = None,
        decoder: Optional = None,
    ):
        """
        Constructs a VQVAE with FC branch

        Parameters
        ----------
        emb_shape : tuple
            Embedding tensor shape
        vq_args : dict
            Additional arguments for the Vector Quantization layer
        num_class : int
            Number of output classes for fc layers
        input_shape : tuple
            Input tensor shape
        output_shape : tuple
            Output tensor shape
        fc_input_type : str
            Input type for the fc layers;
            vqvec: quantized vector, vqind: quantized index, vqindhist: quantized index histogram
        encoder_args : dict
            Additional arguments for encoder
        decoder_args : dict
            Additional arguments for decoder
        fc_args : dict
            Additional arguments for fc layers
        encoder : encoder module
            (Optional) Custom encoder module
        decoder : decoder module
            (Optional) Custom decoder module
        """
        super().__init__(emb_shape, vq_args, input_shape, output_shape, encoder_args, decoder_args, encoder, decoder)
        if fc_args is None:
            fc_args = {'num_layers': 2, 'num_features': 1000}
        if fc_input_type == 'vqind':
            fc_args['in_channels'] = np.prod(self.emb_shape[1:])
        elif fc_input_type == 'vqindhist':
            fc_args['in_channels'] = self.vq_args['num_embeddings']
        else:
            fc_args['in_channels'] = np.prod(self.emb_shape) * self.vq_args['channel_split']
        fc_args['out_channels'] = num_class
        self.fc_layer = FCblock(**fc_args)
        self.fc_loss = None
        self.fc_input_type = fc_input_type

    def forward(self, x: Tensor, output_layer: str = 'decoder') -> tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        if output_layer == 'encoder':
            return encoded
        (
            self.vq_loss,
            quantized,
            self.perplexity,
            _,
            encoding_indices,
            index_histogram,
            softmax_histogram,
        ) = self.vq_layer(encoded)
        if output_layer == 'vqvec':
            return quantized
        elif output_layer == 'vqind':
            return encoding_indices
        elif output_layer == 'vqindhist':
            return index_histogram

        x = self.decoder(quantized)

        if self.fc_input_type == 'vqvec':
            fcout = self.fc_layer(quantized.reshape(quantized.size(0), -1))
        elif self.fc_input_type == 'vqind':
            fcout = self.fc_layer(encoding_indices.reshape(encoding_indices.size(0), -1))
        elif self.fc_input_type == 'vqindhist':
            fcout = self.fc_layer(softmax_histogram)
        else:
            fcout = self.fc_layer(encoded.reshape(encoded.size(0), -1))
        return x, fcout

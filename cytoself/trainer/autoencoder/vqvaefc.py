from typing import Optional
from torch import Tensor
import numpy as np
from cytoself.components.blocks.fc_block import FCblock
from cytoself.trainer.autoencoder.vqvae import VQVAE


class VQVAEFC(VQVAE):
    """
    Vector Quantized Variational Autoencoder with an FC block
    """

    def __init__(
        self,
        input_shape: tuple,
        emb_shape: tuple,
        output_shape: tuple,
        vq_args: dict,
        num_class: int,
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
        input_shape : tuple
            Input tensor shape
        emb_shape : tuple
            Embedding tensor shape
        output_shape : tuple
            Output tensor shape
        vq_args : dict
            Additional arguments for the Vector Quantization layer
        num_class : int
            Number of output classes for fc layers
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
        super().__init__(input_shape, emb_shape, output_shape, vq_args, encoder_args, decoder_args, encoder, decoder)
        if fc_args is None:
            fc_args = {'num_layers': 2, 'num_features': 1000}
        if fc_input_type == 'vqind':
            fc_args['in_channels'] = np.prod(emb_shape[1:])
        elif fc_input_type == 'vqindhist':
            fc_args['in_channels'] = vq_args['num_embeddings']
        else:
            fc_args['in_channels'] = np.prod(emb_shape)
        fc_args['out_channels'] = num_class
        self.fc_layer = FCblock(**fc_args)
        self.fc_loss = None
        self.fc_connection = fc_input_type

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        (
            self.vq_loss,
            quantized,
            self.perplexity,
            self.encoding_onehot,
            self.encoding_indices,
            self.index_histogram,
        ) = self.vq_layer(encoded)
        x = self.decoder(quantized)

        if self.fc_connection == 'vqvec':
            fcout = self.fc_layer(quantized.view(quantized.size(0), -1))
        elif self.fc_connection == 'vqind':
            fcout = self.fc_layer(self.encoding_indices.view(self.encoding_indices.size(0), -1))
        elif self.fc_connection == 'vqindhist':
            fcout = self.fc_layer(self.index_histogram.view(self.index_histogram.size(0), -1))
        else:
            fcout = self.fc_layer(encoded.view(encoded.size(0), -1))
        return x, fcout

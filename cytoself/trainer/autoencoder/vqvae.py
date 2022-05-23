from typing import Optional
from torch import nn, Tensor

from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.components.layers.vq import VectorQuantizer


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder model
    """

    def __init__(
        self,
        input_shape: tuple,
        emb_shape: tuple,
        output_shape: tuple,
        vq_args: dict,
        encoder_args: Optional[dict] = None,
        decoder_args: Optional[dict] = None,
        encoder: Optional = None,
        decoder: Optional = None,
    ):
        super().__init__()
        if encoder is None:
            encoder = efficientenc_b0
        if decoder is None:
            decoder = DecoderResnet
        if encoder_args is None:
            encoder_args = {'in_channels': input_shape[0], 'out_channels': emb_shape[0]}
        if decoder_args is None:
            decoder_args = {'input_shape': emb_shape, 'output_shape': output_shape}
        self.encoder = encoder(**encoder_args)
        self.decoder = decoder(**decoder_args)
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

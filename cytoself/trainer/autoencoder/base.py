from typing import Optional

from torch import Tensor, nn

from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0


class BaseAE(nn.Module):
    """
    Vanilla Autoencoder model
    """

    def __init__(
        self,
        emb_shape: Optional[tuple[int, int, int]] = None,
        input_shape: Optional[tuple[int, int, int]] = None,
        output_shape: Optional[tuple[int, int, int]] = None,
        encoder_args: Optional[dict] = None,
        decoder_args: Optional[dict] = None,
        encoder: Optional = None,
        decoder: Optional = None,
    ):
        """
        Constructs simple encoder decoder model

        Parameters
        ----------
        emb_shape : tuple
            Embedding tensor shape
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
        super().__init__()
        if encoder is None:
            if encoder_args is None:
                encoder_args = {}
            encoder_args.update({'in_channels': input_shape[0], 'out_channels': emb_shape[0]})
            self.encoder = efficientenc_b0(**encoder_args)
        else:
            self.encoder = encoder

        if decoder is None:
            if decoder_args is None:
                decoder_args = {}
            decoder_args.update({'input_shape': emb_shape, 'output_shape': output_shape})
            self.decoder = DecoderResnet(**decoder_args)
        else:
            self.decoder = decoder

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

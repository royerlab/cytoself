from collections.abc import Collection
from typing import Optional, Sequence, Union

import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from cytoself.trainer.autoencoder.cytoselffull import CytoselfFull
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet

"""
CytoselfLite only has decoder1.
The performance is not as good as CytoselfFull model.
"""


class CytoselfLite(CytoselfFull):
    """
    CytoselfLite model (2-stage encoder & decoder with 2 VQ layers and 2 fc blocks)
    EfficientNet_B0 is used for encoders for the sake of saving computation resource.
    """

    def __init__(
        self,
        emb_shapes: Collection[tuple[int, int]],
        vq_args: Union[dict, Collection[dict]],
        num_class: int,
        input_shape: Optional[tuple[int, int, int]] = None,
        output_shape: Optional[tuple[int, int, int]] = None,
        fc_input_type: str = 'vqvec',
        fc_output_idx: Union[str, Sequence[int]] = 'all',
        encoder_args: Optional[Collection[dict]] = None,
        decoder_args: Optional[Collection[dict]] = None,
        fc_args: Optional[Union[dict, Collection[dict]]] = None,
        encoders: Optional[Collection] = None,
        decoders: Optional[Collection] = None,
    ):
        """
        Construct a cytoself light model

        Parameters
        ----------
        emb_shapes : tuple or list of tuples
            Embedding tensor shape except for channel dim
        vq_args : dict
            Additional arguments for the Vector Quantization layer
        num_class : int
            Number of output classes for fc layers
        input_shape : tuple of int
            Input tensor shape
        output_shape : tuple of int
            Output tensor shape; will be same as input_shape if None.
        fc_input_type : str
            Input type for the fc layers;
            vqvec: quantized vector, vqind: quantized index, vqindhist: quantized index histogram
        fc_output_idx : int or 'all'
            Index of encoder to connect FC layers
        encoder_args : dict
            Additional arguments for encoder
        decoder_args : dict
            Additional arguments for decoder
        fc_args : dict
            Additional arguments for fc layers
        encoders : encoder module
            (Optional) Custom encoder module
        decoders : decoder module
            (Optional) Custom decoder module
        """
        super().__init__(
            emb_shapes,
            vq_args,
            num_class,
            input_shape,
            output_shape,
            fc_input_type,
            fc_output_idx,
            encoder_args,
            decoder_args,
            fc_args,
            encoders,
            decoders,
        )

    def _const_decoders(self, output_shape, decoder_args) -> nn.ModuleList:
        """
        Constructs a Module list of decoders

        Parameters
        ----------
        output_shape : tuple
            Output tensor shape
        decoder_args : dict
            Additional arguments for decoder

        Returns
        -------
        nn.ModuleList

        """
        if decoder_args is None:
            decoder_args = [{}] * len(self.emb_shapes)

        decoders = nn.ModuleList()
        for i, shp in enumerate(self.emb_shapes):
            if i == 0:
                shp = (sum(i[0] for i in self.emb_shapes),) + shp[1:]
                decoder_args[i].update(
                    {
                        'input_shape': shp,
                        'output_shape': output_shape if i == 0 else self.emb_shapes[i - 1],
                        'linear_output': i == 0,
                    }
                )
                decoders.append(DecoderResnet(**decoder_args[i]))
            else:
                decoders.append(nn.Module())
        return decoders

    def _connect_decoders(self, encoded_list):
        decoding_list = []
        for i, (encd, dec) in enumerate(zip(encoded_list[::-1], self.decoders[::-1])):
            if i < len(self.decoders) - 1:
                decoding_list.append(resize(encd, self.emb_shapes[0][1:], interpolation=InterpolationMode.NEAREST))
            else:
                decoding_list.append(encd)
                decoded_final = dec(torch.cat(decoding_list, 1))
        return decoded_final

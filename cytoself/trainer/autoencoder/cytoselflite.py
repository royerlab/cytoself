import inspect
from copy import copy
from typing import Optional, Union
from collections.abc import Collection

import torch
from torch import nn, Tensor
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from cytoself.components.blocks.fc_block import FCblock
from cytoself.components.layers.vq import VectorQuantizer
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0

"""
Cytoself by default uses a split EfficientNet B0 model as two encoders.
Therefore, it needs two sets of block arguments for generating two EfficientNets.
"""
default_block_args = [
    # block arguments for the first encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 1,
                'kernel': 3,
                'stride': 1,
                'input_channels': 32,
                'out_channels': 16,
                'num_layers': 1,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 16,
                'out_channels': 24,
                'num_layers': 2,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 1,
                'input_channels': 24,
                'out_channels': 40,
                'num_layers': 2,
            },
        ]
    },
    # block arguments for the second encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 40,
                'out_channels': 80,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,  # 1 in the original
                'input_channels': 80,
                'out_channels': 112,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,
                'input_channels': 112,
                'out_channels': 192,
                'num_layers': 4,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 1,
                'input_channels': 192,
                'out_channels': 320,
                'num_layers': 1,
            },
        ]
    },
]


def length_checker(var1: Collection, var2: Collection):
    """
    Checks if two variables have the same length.
    Specifically to check if the number of parameters match with the number of encoders.

    Parameters
    ----------
    var1 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders
    var2 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders

    Returns
    -------
    none

    """
    if len(var1) != len(var2):
        raise ValueError(f'{var1=}'.split('=')[0] + ' and ' + f'{var2=}'.split('=')[0] + ' must be the same length.')


def duplicate_kwargs(arg1: Collection, arg2: Collection):
    """
    Duplicate arg1 by the length of arg2.

    Parameters
    ----------
    arg1 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders
    arg2 : Collection of arguments
        Encoder dependent argument whose number should match with the number of encoders

    Returns
    -------
    list

    """
    if isinstance(arg1, dict):
        arg1 = [arg1] * len(arg2)
    else:
        if len(arg1) != len(arg2):
            raise ValueError(f'The length of arg1 {len(arg1)} must match with that of arg2 {len(arg2)}.')
    return arg1


def calc_emb_dim(vq_args, emb_shapes):
    vq_args = copy(vq_args)
    emb_shapes_out = []
    for i, varg in enumerate(vq_args):
        if 'embedding_dim' not in varg:
            raise ValueError('embedding_dim is required for vq_args.')
        if 'channel_split' not in varg:
            varg['channel_split'] = inspect.signature(VectorQuantizer).parameters['channel_split'].default
        emb_shapes_out.append((varg['embedding_dim'] * varg['channel_split'],) + tuple(emb_shapes[i]))
    return vq_args, tuple(emb_shapes_out)


class CytoselfLite(nn.Module):
    """
    CytoselfLite original model (2-stage encoder & decoder with 2 VQ layers and 2 fc blocks)
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
        super().__init__()
        # Check vq_args and emb_shapes and compute emb_ch_splits
        vq_args = duplicate_kwargs(vq_args, emb_shapes)
        vq_args, emb_shapes = calc_emb_dim(vq_args, emb_shapes)
        self.emb_shapes = emb_shapes

        # Construct encoders (from the one close to input data to the one to far from)
        if encoders is None:
            self.encoders = self._const_encoders(input_shape, encoder_args)
        else:
            self.encoders = encoders

        # Construct decoders (shallow to deep)
        if decoders is None:
            if output_shape is None:
                output_shape = input_shape
            self.decoders = self._const_decoders(output_shape, decoder_args)
        else:
            self.decoders = decoders

        # Construct VQ layers (same order as encoders)
        self.vq_layers = nn.ModuleList()
        for i, varg in enumerate(vq_args):
            self.vq_layers.append(VectorQuantizer(**varg))
        self.vq_loss = None
        self.perplexity = None

        # Construct fc blocks (same order as encoders)
        if fc_args is None:
            fc_args = {'num_layers': 2, 'num_features': 1000}
        fc_args = duplicate_kwargs(fc_args, emb_shapes)

        self.fc_layers = nn.ModuleList()
        for i, shp in enumerate(emb_shapes):
            arg = fc_args[i]
            if fc_input_type == 'vqind':
                arg['in_channels'] = np.prod(shp[1:])
            elif fc_input_type == 'vqindhist':
                arg['in_channels'] = vq_args[i]['num_embeddings']
            else:
                arg['in_channels'] = np.prod(shp)
            arg['out_channels'] = num_class
            self.fc_layers.append(FCblock(**arg))
        self.fc_loss = None
        self.fc_input_type = fc_input_type

    def _const_encoders(self, input_shape, encoder_args) -> nn.ModuleList:
        """
        Constructs a Module list of encoders

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape
        encoder_args : dict
            Additional arguments for encoder

        Returns
        -------
        nn.ModuleList

        """
        if encoder_args is None:
            encoder_args = default_block_args
        length_checker(self.emb_shapes, encoder_args)

        encoders = nn.ModuleList()
        for i, shp in enumerate(self.emb_shapes):
            encoder_args[i].update(
                {
                    'in_channels': input_shape[0] if i == 0 else self.emb_shapes[i - 1][0],
                    'out_channels': shp[0],
                    'first_layer_stride': 2 if i == 0 else 1,
                }
            )
            encoders.append(efficientenc_b0(**encoder_args[i]))
        return encoders

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

    def forward(self, x: Tensor, output_layer: str = 'decoder0') -> tuple[Tensor, Tensor]:
        """
        Cytoselflite model consists of encoders & decoders such as:
        encoder1 -> vq layer1 -> encoder2 -> vq layer2 -> ... -> decoder2 -> decoder1
        The order in the Module list of encoders and decoders is always 1 -> 2 -> ...

        Parameters
        ----------
        x : Tensor
            Image data
        output_layer : str
            Name of layer + index (integer) as the exit layer.
            This is used to indicate the exit layer for output embeddings.

        Returns
        -------
        A list of Tensors

        """
        self.vq_loss = {}
        self.perplexity = {}
        out_layer_name, out_layer_idx = output_layer[:-1], int(output_layer[-1]) - 1

        fc_outs = []
        encoded_list = []
        for i, enc in enumerate(self.encoders):
            encoded = enc(x)
            if out_layer_name == 'encoder' and i == out_layer_idx:
                return encoded

            (
                vq_loss,
                quantized,
                perplexity,
                _,
                _encoding_indices,
                _index_histogram,
                softmax_histogram,
            ) = self.vq_layers[i](encoded)
            if i == out_layer_idx:
                if out_layer_name == 'vqvec':
                    return quantized
                elif out_layer_name == 'vqind':
                    return _encoding_indices
                elif out_layer_name == 'vqindhist':
                    return _index_histogram

            if self.fc_input_type == 'vqvec':
                fcout = self.fc_layers[i](quantized.view(quantized.size(0), -1))
            elif self.fc_input_type == 'vqind':
                fcout = self.fc_layers[i](_encoding_indices.view(_encoding_indices.size(0), -1))
            elif self.fc_input_type == 'vqindhist':
                fcout = self.fc_layers[i](softmax_histogram)
            else:
                fcout = self.fc_layers[i](encoded.view(encoded.size(0), -1))

            fc_outs.append(fcout)
            encoded_list.append(encoded)
            self.vq_loss[f'vq{i + 1}'] = vq_loss
            self.perplexity[f'perplexity{i + 1}'] = perplexity
            x = encoded

        decoding_list = []
        for i, (encd, dec) in enumerate(zip(encoded_list[::-1], self.decoders[::-1])):
            if i < len(self.decoders) - 1:
                decoding_list.append(resize(encd, self.emb_shapes[0][1:], interpolation=InterpolationMode.NEAREST))
            else:
                decoding_list.append(encd)
                decoded_final = dec(torch.cat(decoding_list, 1))

        # TODO: implement mse_layer for decoder2 and encoded1

        return tuple([decoded_final] + fc_outs)

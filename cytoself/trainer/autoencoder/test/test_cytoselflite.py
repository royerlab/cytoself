import torch

from cytoself.trainer.autoencoder.cytoselflite import CytoselfLite
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet


def test_cytoselflite():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((64, 25, 25), (64, 4, 4))
    for t in ['vqvec', 'vqind', 'vqindhist', 'enc']:
        model = CytoselfLite(
            emb_shape,
            {'num_embeddings': 7},
            3,
            input_shape,
            input_shape,
            t,
            fc_args={'num_layers': 1, 'num_features': 10},
        )
        model.to(device)
        input_data = torch.randn((1,) + input_shape).to(device)
        out = model(input_data)
        assert len(out) == 3
        assert out[0].shape == input_data.shape
        assert out[1].shape == (1, 3)
        assert out[2].shape == (1, 3)


def test_cytoselflite_encoding():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((64, 25, 25), (64, 4, 4))
    model = CytoselfLite(
        emb_shape,
        {'num_embeddings': 7},
        3,
        input_shape,
        input_shape,
        'vqvec',
        fc_args={'num_layers': 1, 'num_features': 10},
    )
    model.to(device)
    input_data = torch.randn((1,) + input_shape).to(device)
    for i in range(2):
        out = model(input_data, f'encoder{i + 1}')
        assert out.shape == (1,) + emb_shape[i]
        out = model(input_data, f'vqvec{i + 1}')
        assert out.shape == (1,) + emb_shape[i]
        out = model(input_data, f'vqind{i + 1}')
        assert out.shape == (1,) + emb_shape[i][1:]
        out = model(input_data, f'vqindhist{i + 1}')
        assert out.shape == (1, 7)


def test_cytoselflite_custom():
    input_shape, emb_shape = (2, 100, 100), ((64, 25, 25), (64, 4, 4))
    encoders = [
        efficientenc_b0(in_channels=input_shape[0], out_channels=emb_shape[0][0]),
        efficientenc_b0(in_channels=emb_shape[0][0], out_channels=emb_shape[1][0]),
    ]
    decoders = [
        DecoderResnet(input_shape=emb_shape[0], output_shape=input_shape),
        DecoderResnet(input_shape=emb_shape[1], output_shape=emb_shape[0]),
    ]
    model = CytoselfLite(emb_shape, {'num_embeddings': 7}, 3, encoders=encoders, decoders=decoders)
    for i in range(2):
        assert model.encoders[i] == encoders[i]
        assert model.decoders[i] == decoders[i]

import pytest
import torch

from cytoself.trainer.autoencoder.cytoselffull import CytoselfFull, calc_emb_dim, duplicate_kwargs, length_checker
from cytoself.trainer.autoencoder.decoders.resnet2d import DecoderResnet
from cytoself.trainer.autoencoder.encoders.efficientenc2d import efficientenc_b0


def test_duplicate_kwargs():
    arg1, arg2 = {'a': 1, 'b': 2}, (64, 64)
    arg_out = duplicate_kwargs(arg1, arg2)
    assert len(arg_out) == len(arg2)
    assert all([a == arg1 for a in arg_out])
    with pytest.raises(ValueError):
        duplicate_kwargs((arg1,), arg2)


def test_calc_emb_dim():
    vq_args = [{'num_embeddings': 7}]
    emb_shapes = [[4, 4]]
    with pytest.raises(ValueError):
        _, _ = calc_emb_dim(vq_args, emb_shapes)

    vq_args = [{'num_embeddings': 7, 'embedding_dim': 64}]
    vq_args_out, emb_shapes_out = calc_emb_dim(vq_args, emb_shapes)
    assert 'channel_split' in vq_args_out[0]
    assert vq_args_out[0]['channel_split'] == 1
    assert emb_shapes_out == ((64, 4, 4),)

    vq_args = [{'num_embeddings': 7, 'embedding_dim': 64, 'channel_split': 2}]
    vq_args_out, emb_shapes_out = calc_emb_dim(vq_args, emb_shapes)
    assert emb_shapes_out == ((64 * 2, 4, 4),)


def test_length_checker():
    with pytest.raises(ValueError):
        length_checker([(4, 4)], [{'a': 1}, {'a': 1}])


def test_cytoselffull():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((25, 25), (4, 4))
    for t in ['vqvec', 'vqind', 'vqindhist', 'enc']:
        model = CytoselfFull(
            emb_shape,
            {'num_embeddings': 7, 'embedding_dim': 64},
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


def test_cytoselffull_fc2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((25, 25), (4, 4))
    for fci in [1, 2]:
        for t in ['vqvec', 'vqind', 'vqindhist', 'enc']:
            model = CytoselfFull(
                emb_shape,
                {'num_embeddings': 7, 'embedding_dim': 64},
                3,
                input_shape,
                input_shape,
                t,
                [fci],
                fc_args={'num_layers': 1, 'num_features': 10},
            )
            model.to(device)
            input_data = torch.randn((1,) + input_shape).to(device)
            out = model(input_data)
            assert len(out) == 2
            assert out[0].shape == input_data.shape
            assert out[1].shape == (1, 3)


def test_cytoselffull_nofc():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((25, 25), (4, 4))
    for t in ['vqvec', 'vqind', 'vqindhist', 'enc']:
        model = CytoselfFull(
            emb_shape,
            {'num_embeddings': 7, 'embedding_dim': 64},
            3,
            input_shape,
            input_shape,
            t,
            [],
            fc_args={'num_layers': 1, 'num_features': 10},
        )
        model.to(device)
        input_data = torch.randn((1,) + input_shape).to(device)
        out = model(input_data)
        assert len(out) == 1
        assert out[0].shape == input_data.shape


def test_cytoselffull_encoding():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), ((25, 25), (4, 4))
    model = CytoselfFull(
        emb_shape,
        {'num_embeddings': 7, 'embedding_dim': 64},
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
        assert out.shape == (1, 64) + emb_shape[i]
        out = model(input_data, f'vqvec{i + 1}')
        assert out.shape == (1, 64) + emb_shape[i]
        out = model(input_data, f'vqind{i + 1}')
        assert out.shape == (1, 1) + emb_shape[i]
        out = model(input_data, f'vqindhist{i + 1}')
        assert out.shape == (1, 7)


def test_cytoselffull_custom():
    input_shape, emb_shape = (2, 100, 100), ((25, 25), (4, 4))
    encoders = [
        efficientenc_b0(in_channels=input_shape[0], out_channels=emb_shape[0][0]),
        efficientenc_b0(in_channels=emb_shape[0][0], out_channels=emb_shape[1][0]),
    ]
    decoders = [
        DecoderResnet(input_shape=emb_shape[0], output_shape=input_shape),
        DecoderResnet(input_shape=emb_shape[1], output_shape=emb_shape[0]),
    ]
    model = CytoselfFull(emb_shape, {'num_embeddings': 7, 'embedding_dim': 64}, 3, encoders=encoders, decoders=decoders)
    for i in range(2):
        assert model.encoders[i] == encoders[i]
        assert model.decoders[i] == decoders[i]

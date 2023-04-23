import pytest
import torch

from cytoself.components.layers.vq import VectorQuantizer, split_channel, unsplit_channel


def test_split_channel():
    z = torch.arange(1000).view(4, 10, 5, 5)
    zz = split_channel(z, 2, 5)
    assert (z[:, :5, ...] == zz[..., ::2]).all()
    assert (z[:, 5:, ...] == zz[..., 1::2]).all()
    with pytest.raises(ValueError):
        split_channel(z, 3, 5)


def test_unsplit_channel():
    z = torch.arange(1000).view(4, 10, 5, 5)
    zz = unsplit_channel(split_channel(z, 2, 5), 2)
    assert (z == zz).all()


def test_VectorQuantizer():
    embedding_dim = 10
    num_embeddings = 3
    vq = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost=1.0)
    assert vq.codebook.weight.shape == (num_embeddings, embedding_dim)

    data = torch.randn((5, embedding_dim, 4, 4), dtype=torch.float32)
    out = vq(data)
    assert len(out[0]) == 4
    for k, v in out[0].items():
        assert len(v.shape) == 0
    assert out[1].shape == data.shape
    assert len(out[2].shape) == 0
    assert out[3].max() == 1
    assert out[3].shape == (data.shape[0], num_embeddings) + data.shape[2:]
    assert out[4].shape == data.shape[:1] + (1,) + data.shape[2:]
    assert out[5].shape == (data.shape[0], num_embeddings)
    assert out[6].shape == (data.shape[0], num_embeddings)
    assert all(torch.round(out[5].sum(axis=1), decimals=3) == 16)
    assert (out[5].int() == out[5]).all()


def test_VectorQuantizer_chsplit():
    embedding_dim = 10
    num_embeddings = 3
    vq = VectorQuantizer(embedding_dim, num_embeddings, channel_split=2)
    assert vq.codebook.weight.shape == (num_embeddings, embedding_dim)

    data = torch.randn((5, embedding_dim * 2, 4, 4), dtype=torch.float32)
    out = vq(data)
    assert len(out[0]) == 4
    for k, v in out[0].items():
        assert len(v.shape) == 0
    assert out[1].shape == data.shape
    assert len(out[2].shape) == 0
    assert out[3].max() == 1
    assert out[3].shape == (data.shape[0], num_embeddings * 2) + data.shape[2:]
    assert out[4].shape == data.shape[:1] + (2,) + data.shape[2:]
    assert out[5].shape == (data.shape[0], num_embeddings)
    assert out[6].shape == (data.shape[0], num_embeddings)
    assert all(torch.round(out[5].sum(axis=1), decimals=3) == 16 * 2)
    assert (out[5].int() == out[5]).all()

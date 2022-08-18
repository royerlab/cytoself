import torch
from cytoself.components.layers.vq import VectorQuantizer


def test_VectorQuantizer():
    embedding_dim = 10
    num_embeddings = 3
    vq = VectorQuantizer(embedding_dim, num_embeddings, 1.0)
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
    assert out[4].shape == data.shape[:1] + data.shape[2:]
    assert out[5].shape == (data.shape[0], num_embeddings)
    assert out[6].shape == (data.shape[0], num_embeddings)
    assert all(torch.round(out[5].sum(axis=1), decimals=3) == 16)
    assert (out[5].int() == out[5]).all()

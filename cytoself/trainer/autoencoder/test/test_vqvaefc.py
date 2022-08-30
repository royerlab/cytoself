import torch

from cytoself.trainer.autoencoder.vqvaefc import VQVAEFC


def test_VQVAEFC():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (4, 4)
    for t in ['vqvec', 'vqind', 'vqindhist', 'enc']:
        model = VQVAEFC(
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
        assert len(out) == 2
        assert out[0].shape == input_data.shape
        assert out[1].shape == (1, 3)


def test_VQVAEFC_outputlayer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (4, 4)
    model = VQVAEFC(
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
    assert model(input_data, 'encoder').shape == (len(input_data), 64) + emb_shape
    assert model(input_data, 'vqvec').shape == (len(input_data), 64) + emb_shape
    assert model(input_data, 'vqind').shape == (len(input_data), 1) + emb_shape
    assert model(input_data, 'vqindhist').shape == (len(input_data), 7)

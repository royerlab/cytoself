import torch

from cytoself.trainer.autoencoder.vqvae import VQVAE


def test_VQVAE():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (64, 4, 4)
    model = VQVAE(emb_shape, {'num_embeddings': 7}, input_shape, input_shape)
    model.to(device)
    input_data = torch.randn((1,) + input_shape).to(device)
    out = model(input_data)
    assert out.shape == input_data.shape
    assert len(model.vq_loss['loss'].shape) == 0
    assert len(model.perplexity.shape) == 0
    assert model.encoding_onehot.max() == 1
    assert model.encoding_indices.shape == input_data.shape[:1] + emb_shape[1:]

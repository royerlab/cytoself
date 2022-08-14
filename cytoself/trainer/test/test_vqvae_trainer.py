from os.path import join

import pytest
import torch

from cytoself.test_util.test_parameters import add_default_model_args
from cytoself.trainer.vqvae_trainer import VQVAETrainer


@pytest.fixture(scope='module')
def vqvae_trainer(basepath):
    model_args = {
        'input_shape': (1, 32, 32),
        'emb_shape': (16, 16, 16),
        'output_shape': (1, 32, 32),
        'vq_args': {'num_embeddings': 7},
    }
    model_args = add_default_model_args(model_args)
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    return VQVAETrainer(model_args, train_args, homepath=basepath)


def test_vqvae_trainer_fit(vqvae_trainer, opencell_datamgr_vanilla, basepath):
    vqvae_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(vqvae_trainer.losses['train_loss']) == vqvae_trainer.train_args['max_epoch']
    assert min(vqvae_trainer.losses['train_loss']) < torch.inf
    assert min(vqvae_trainer.losses['train_vq_loss']) < torch.inf

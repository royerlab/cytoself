from os.path import join

import pytest
import torch

from cytoself.test_util.test_parameters import add_default_model_args
from cytoself.trainer.vqvaefc_trainer import VQVAEFCTrainer


@pytest.fixture(scope='module')
def vqvaefc_trainer(basepath):
    model_args = {
        'input_shape': (1, 32, 32),
        'emb_shape': (16, 16),
        'output_shape': (1, 32, 32),
        'vq_args': {'num_embeddings': 7, 'embedding_dim': 16},
        'num_class': 3,
    }
    model_args = add_default_model_args(model_args)
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    return VQVAEFCTrainer(train_args, homepath=basepath, model_args=model_args)


def test_vqvaefc_trainer_fit(vqvaefc_trainer, opencell_datamgr_vanilla, basepath):
    vqvaefc_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(vqvaefc_trainer.history['train_loss']) == vqvaefc_trainer.train_args['max_epoch']
    assert min(vqvaefc_trainer.history['train_loss']) < torch.inf
    assert min(vqvaefc_trainer.history['train_vq_loss']) < torch.inf

from os.path import join

import pytest
import torch

from cytoself.test_util.test_parameters import CYTOSELF_MODEL_ARGS
from cytoself.trainer.cytoselflight_trainer import CytoselfLiteTrainer


@pytest.fixture(scope='module')
def cytoselflight_trainer(basepath):
    train_args = {'lr': 1e-6, 'max_epochs': 2}
    return CytoselfLiteTrainer(CYTOSELF_MODEL_ARGS, train_args, homepath=basepath)


def test_cytoselflight_trainer_fit(cytoselflight_trainer, opencell_datamgr_vanilla, basepath):
    cytoselflight_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(cytoselflight_trainer.losses['train_loss']) == cytoselflight_trainer.train_args['max_epochs']
    assert min(cytoselflight_trainer.losses['train_loss']) < torch.inf
    assert min(cytoselflight_trainer.losses['train_vq_loss1']) < torch.inf
    assert min(cytoselflight_trainer.losses['train_vq_loss2']) < torch.inf
    assert min(cytoselflight_trainer.losses['train_fc_loss1']) < torch.inf
    assert min(cytoselflight_trainer.losses['train_fc_loss2']) < torch.inf
    assert all([loss.requires_grad is False for loss in cytoselflight_trainer.model.vq_loss])
    assert all([loss.requires_grad is False for loss in cytoselflight_trainer.model.fc_loss])


def test_cytoselflight_trainer_embeddings(cytoselflight_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselflight_trainer.infer_embeddings(None)

    out = cytoselflight_trainer.infer_embeddings(opencell_datamgr_vanilla.test_loader)
    assert out[0].shape == (len(opencell_datamgr_vanilla.test_dataset),) + CYTOSELF_MODEL_ARGS['emb_shapes'][1]

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselflight_trainer.infer_embeddings(d['image'].numpy())
    assert out.shape == (opencell_datamgr_vanilla.batch_size,) + CYTOSELF_MODEL_ARGS['emb_shapes'][1]


def test_cytoselflight_trainer_reconstruction(cytoselflight_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselflight_trainer.infer_embeddings(None)

    out = cytoselflight_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselflight_trainer.infer_reconstruction(d['image'].numpy())
    assert out.shape == (opencell_datamgr_vanilla.batch_size,) + opencell_datamgr_vanilla.test_dataset.data.shape[1:]

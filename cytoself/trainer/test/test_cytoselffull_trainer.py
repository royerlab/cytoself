from copy import copy

import pytest
import torch

from cytoself.test_util.test_parameters import CYTOSELF_MODEL_ARGS
from cytoself.trainer.cytoselffull_trainer import CytoselfFullTrainer


@pytest.mark.parametrize('fc_output_idx', [[1], [2], []])
def test_cytoselffull_xfc(basepath, opencell_datamgr_vanilla, fc_output_idx):
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    model_args = copy(CYTOSELF_MODEL_ARGS)
    model_args['fc_output_idx'] = fc_output_idx
    trainer = CytoselfFullTrainer(train_args, homepath=basepath, model_args=model_args)
    trainer.fit(opencell_datamgr_vanilla)
    if len(fc_output_idx) > 0:
        assert f'train_fc{fc_output_idx[0]}_loss' in trainer.history
    else:
        assert 'train_fc1_loss' not in trainer.history


def test_cytoselffull_trainer_fit(cytoselffull_trainer, opencell_datamgr_vanilla, basepath):
    cytoselffull_trainer.fit(opencell_datamgr_vanilla)
    assert len(cytoselffull_trainer.history['train_loss']) == cytoselffull_trainer.train_args['max_epoch']
    assert min(cytoselffull_trainer.history['train_loss']) < torch.inf
    assert min(cytoselffull_trainer.history['train_vq1_loss']) < torch.inf
    assert min(cytoselffull_trainer.history['train_vq2_loss']) < torch.inf
    assert min(cytoselffull_trainer.history['train_fc1_loss']) < torch.inf
    assert min(cytoselffull_trainer.history['train_fc2_loss']) < torch.inf
    assert min(cytoselffull_trainer.history['train_reconstruction2_loss']) < torch.inf
    assert all(
        [all([l1.requires_grad is False for l1 in l0.values()]) for l0 in cytoselffull_trainer.model.vq_loss.values()]
    )
    assert all([loss.requires_grad is False for loss in cytoselffull_trainer.model.fc_loss.values()])


def test_cytoselffull_trainer_embeddings(cytoselffull_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselffull_trainer.infer_embeddings(None)

    out = cytoselffull_trainer.infer_embeddings(opencell_datamgr_vanilla.test_loader)
    assert (
        out[0].shape
        == (len(opencell_datamgr_vanilla.test_loader.dataset), CYTOSELF_MODEL_ARGS['vq_args']['embedding_dim'])
        + CYTOSELF_MODEL_ARGS['emb_shapes'][1]
    )

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselffull_trainer.infer_embeddings(d['image'].numpy())
    assert (
        out.shape
        == (opencell_datamgr_vanilla.test_loader.batch_size, CYTOSELF_MODEL_ARGS['vq_args']['embedding_dim'])
        + CYTOSELF_MODEL_ARGS['emb_shapes'][1]
    )


def test_cytoselffull_trainer_reconstruction(cytoselffull_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselffull_trainer.infer_embeddings(None)

    out = cytoselffull_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_loader.dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselffull_trainer.infer_reconstruction(d['image'].numpy())
    assert (
        out.shape
        == (opencell_datamgr_vanilla.test_loader.batch_size,)
        + opencell_datamgr_vanilla.test_loader.dataset.data.shape[1:]
    )

from os.path import join

import pytest
import torch

from cytoself.test_util.test_parameters import CYTOSELF_MODEL_ARGS


def test_cytoselffull_trainer_fit(cytoselffull_trainer, opencell_datamgr_vanilla, basepath):
    cytoselffull_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
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
        == (len(opencell_datamgr_vanilla.test_dataset), CYTOSELF_MODEL_ARGS['vq_args']['embedding_dim'])
        + CYTOSELF_MODEL_ARGS['emb_shapes'][1]
    )

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselffull_trainer.infer_embeddings(d['image'].numpy())
    assert (
        out.shape
        == (opencell_datamgr_vanilla.batch_size, CYTOSELF_MODEL_ARGS['vq_args']['embedding_dim'])
        + CYTOSELF_MODEL_ARGS['emb_shapes'][1]
    )


def test_cytoselffull_trainer_reconstruction(cytoselffull_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselffull_trainer.infer_embeddings(None)

    out = cytoselffull_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselffull_trainer.infer_reconstruction(d['image'].numpy())
    assert out.shape == (opencell_datamgr_vanilla.batch_size,) + opencell_datamgr_vanilla.test_dataset.data.shape[1:]

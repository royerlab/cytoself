from os.path import join

import pytest
import torch


def test_cytoselflite_trainer_fit(cytoselflite_trainer, opencell_datamgr_vanilla, basepath):
    cytoselflite_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(cytoselflite_trainer.history['train_loss']) == cytoselflite_trainer.train_args['max_epoch']
    assert min(cytoselflite_trainer.history['train_loss']) < torch.inf
    assert min(cytoselflite_trainer.history['train_vq1_loss']) < torch.inf
    assert min(cytoselflite_trainer.history['train_vq2_loss']) < torch.inf
    assert min(cytoselflite_trainer.history['train_fc1_loss']) < torch.inf
    assert min(cytoselflite_trainer.history['train_fc2_loss']) < torch.inf
    assert 'train_reconstruction2_loss' not in cytoselflite_trainer.history
    assert len(cytoselflite_trainer.model.mse_loss) == 1
    assert all(
        [all([l1.requires_grad is False for l1 in l0.values()]) for l0 in cytoselflite_trainer.model.vq_loss.values()]
    )
    assert all([loss.requires_grad is False for loss in cytoselflite_trainer.model.fc_loss.values()])


def test_cytoselflite_trainer_reconstruction(cytoselflite_trainer, opencell_datamgr_vanilla, basepath):
    with pytest.raises(ValueError):
        cytoselflite_trainer.infer_embeddings(None)

    out = cytoselflite_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_loader.dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = cytoselflite_trainer.infer_reconstruction(d['image'].numpy())
    assert (
        out.shape
        == (opencell_datamgr_vanilla.test_loader.batch_size,)
        + opencell_datamgr_vanilla.test_loader.dataset.data.shape[1:]
    )

import os
from copy import deepcopy
from os.path import exists, join, splitext

import pandas as pd
import pytest
import torch
from natsort import natsorted


def test_vanilla_ae_trainer_no_model_args():
    from cytoself.trainer.vanilla_trainer import VanillaAETrainer

    with pytest.raises(ValueError):
        VanillaAETrainer({})


def test_vanilla_ae_trainer_fit(vanilla_ae_trainer, opencell_datamgr_vanilla, basepath):
    vanilla_ae_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(vanilla_ae_trainer.history['train_loss']) == vanilla_ae_trainer.train_args['max_epoch']
    assert min(vanilla_ae_trainer.history['train_loss']) < torch.inf
    assert exists(join(vanilla_ae_trainer.savepath_dict['visualization'], 'training_history.csv'))
    assert pd.read_csv(join(vanilla_ae_trainer.savepath_dict['visualization'], 'training_history.csv')).shape == (2, 4)


def test_infer_reconstruction(vanilla_ae_trainer, opencell_datamgr_vanilla):
    out = vanilla_ae_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_loader.dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = vanilla_ae_trainer.infer_reconstruction(d['image'].numpy())
    assert (
        out.shape
        == (opencell_datamgr_vanilla.test_loader.batch_size,)
        + opencell_datamgr_vanilla.test_loader.dataset.data.shape[1:]
    )


def test_save_load_model(vanilla_ae_trainer):
    vanilla_ae_trainer.save_model(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt')
    vanilla_ae_trainer.save_model(vanilla_ae_trainer.savepath_dict['homepath'], 'test_dict.pt', by_weights=True)
    assert exists(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt')), 'pytorch model was not found.'
    w = deepcopy(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    # load a whole model
    torch.nn.init.zeros_(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    vanilla_ae_trainer.load_model(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt'), by_weights=False)
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()
    # load a weights from a whole model
    torch.nn.init.normal_(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    vanilla_ae_trainer.load_model(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt'), by_weights=True)
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()
    # load a weights directory
    torch.nn.init.uniform_(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    vanilla_ae_trainer.load_model(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test_dict.pt'))
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()


def test_save_load_checkpoint(vanilla_ae_trainer, opencell_datamgr_vanilla, basepath):
    chkp_path = vanilla_ae_trainer.savepath_dict['checkpoints']
    flist0 = [f for f in os.listdir(chkp_path) if f.endswith('.chkp')]
    # trigger save checkpoint
    vanilla_ae_trainer.train_args['max_epoch'] = 101
    vanilla_ae_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'), initial_epoch=100)
    vanilla_ae_trainer.save_checkpoint()
    # check if a new checkpoint is saved
    flist = natsorted([f for f in os.listdir(chkp_path) if f.endswith('.chkp')])
    assert len(flist) > len(flist0)
    # prepare values for control
    w = deepcopy(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    torch.nn.init.zeros_(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    vanilla_ae_trainer.set_optimizer(lr=1e-3)
    vanilla_ae_trainer.history['train_loss'][:] = 1.0

    # load a checkpoint
    vanilla_ae_trainer.load_checkpoint()
    checkpoint = torch.load(join(chkp_path, flist[-1]))
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()
    assert vanilla_ae_trainer.current_epoch == int(splitext(flist[-1])[0].split('_ep')[-1])
    assert (
        vanilla_ae_trainer.optimizer.param_groups[0]['lr']
        == checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    )
    assert all(vanilla_ae_trainer.history == checkpoint['history'])

    # load a checkpoint with a specific epoch number
    torch.nn.init.zeros_(vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight)
    vanilla_ae_trainer.set_optimizer(lr=1e-3)
    vanilla_ae_trainer.history['train_loss'][:] = 1.0
    vanilla_ae_trainer.load_checkpoint(epoch=101)
    checkpoint = torch.load(join(chkp_path, flist[-1]))
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()
    assert vanilla_ae_trainer.current_epoch == int(splitext(flist[-1])[0].split('_ep')[-1])
    assert (
        vanilla_ae_trainer.optimizer.param_groups[0]['lr']
        == checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    )
    assert all(vanilla_ae_trainer.history == checkpoint['history'])

from os.path import join

import pytest

from cytoself.trainer.vanilla_trainer import VanillaAETrainer


@pytest.fixture(scope='module')
def vanilla_ae_trainer_plateau(basepath):
    model_args = {
        'input_shape': (1, 32, 32),
        'emb_shape': (16, 16, 16),
        'output_shape': (1, 32, 32),
    }
    model_args['encoder_args'] = {
        'in_channels': model_args['input_shape'][0],
        'blocks_args': [
            {
                'expand_ratio': 1,
                'kernel': 3,
                'stride': 1,
                'input_channels': 32,
                'out_channels': 16,
                'num_layers': 1,
            }
        ],
        'out_channels': model_args['emb_shape'][0],
    }
    model_args['decoder_args'] = {
        'input_shape': model_args['emb_shape'],
        'num_residual_layers': 1,
        'output_shape': model_args['input_shape'],
    }
    train_args = {
        'lr': 1e-8,
        'max_epoch': 5,
        'reducelr_patience': 1,
        'min_lr': 1e-9,
        'earlystop_patience': 3,
        'optimizer': 'Adam',
    }
    return VanillaAETrainer(train_args, homepath=basepath, model_args=model_args)


def test__reduce_lr_on_plateau(vanilla_ae_trainer_plateau, opencell_datamgr_vanilla, basepath):
    vanilla_ae_trainer_plateau.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert round(vanilla_ae_trainer_plateau.optimizer.param_groups[0]['lr'], 9) == 1e-9
    assert len(vanilla_ae_trainer_plateau.history['train_loss']) == 4

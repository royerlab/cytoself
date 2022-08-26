from os.path import exists

import pandas as pd
import pytest
import torch

from cytoself.trainer.basetrainer import BaseTrainer


@pytest.fixture(scope='module')
def base_trainer(basepath):
    BaseTrainer({}, homepath=basepath, device='cpu')
    # Run second time without device input to cover the specific case
    return BaseTrainer({}, homepath=basepath)


def test_base_trainer_init_(base_trainer, basepath):
    assert base_trainer.model is None
    assert base_trainer.best_model == []
    assert isinstance(base_trainer.history, pd.DataFrame)
    assert base_trainer.history.empty
    assert base_trainer.lr == 0
    assert base_trainer.tb_writer is None
    assert base_trainer.optimizer is None
    assert base_trainer.savepath_dict['homepath'] == basepath
    assert base_trainer.current_epoch == 1


def test_base_trainer__default_train_args(base_trainer):
    base_trainer._default_train_args()
    args = {
        'reducelr_patience': 4,
        'reducelr_increment': 0.1,
        'earlystop_patience': 12,
        'min_lr': 1e-8,
        'max_epoch': 100,
    }
    for key, val in args.items():
        assert base_trainer.train_args[key] == val


def test_calc_loss_one_batch(base_trainer):
    assert base_trainer.calc_loss_one_batch(torch.ones(5) * 3, torch.ones(5))['loss'] == torch.ones(1) * 4


def test_record_metrics(base_trainer):
    base_trainer.record_metrics(pd.DataFrame([{'train_loss': 1.0}]))
    assert (base_trainer.history['train_loss'] == [1.0]).all()
    base_trainer.record_metrics([pd.DataFrame([{'train_loss': 2.0}]), pd.DataFrame([{'val_loss': 3.0}])])
    assert base_trainer.history['train_loss'].to_list() == [1.0, 2.0]
    assert base_trainer.history['val_loss'].to_list() == [0, 3.0]


def test__aggregate_metrics(base_trainer):
    dict_list = [{f'loss{i}': j for i in range(3)} for j in range(10)]
    output = base_trainer._aggregate_metrics(dict_list, 'test')
    assert isinstance(output, pd.DataFrame)
    assert output.shape == (1, 3)
    assert all(['test' in c for c in output.columns])
    assert (output.values == 4.5).all()


def test_set_optimizer(base_trainer, basepath):
    with pytest.raises(ValueError):
        base_trainer.set_optimizer()

    from cytoself.test_util.test_parameters import add_default_model_args
    from cytoself.trainer.vanilla_trainer import VanillaAETrainer

    model_args = {
        'input_shape': (1, 32, 32),
        'emb_shape': (16, 16, 16),
        'output_shape': (1, 32, 32),
    }
    model_args = add_default_model_args(model_args)
    train_args = {'lr': 1e-6, 'max_epoch': 2, 'optimizer': 'SGD'}
    with pytest.raises(ValueError):
        VanillaAETrainer(model_args, train_args, homepath=basepath)


def test_enable_tensorboard(base_trainer):
    base_trainer.savepath_dict['tb_logs'] = 'dummy'
    with pytest.warns(UserWarning):
        base_trainer.enable_tensorboard()
    assert exists(base_trainer.savepath_dict['tb_logs'])


def test_init_savepath(base_trainer):
    base_trainer.init_savepath()
    for key, val in base_trainer.savepath_dict.items():
        assert exists(val), key + ' path was not found.'


def test_train_one_epoch(base_trainer):
    with pytest.raises(ValueError):
        base_trainer.train_one_epoch(None)


def test_calc_val_loss(base_trainer):
    with pytest.raises(ValueError):
        base_trainer.calc_val_loss(None)


def test__reduce_lr_on_plateau(base_trainer):
    with pytest.raises(ValueError):
        base_trainer._reduce_lr_on_plateau(2)


def test_fit(base_trainer):
    with pytest.raises(ValueError):
        base_trainer.fit(None)


def test_infer_embeddings(base_trainer):
    with pytest.raises(ValueError):
        base_trainer.infer_embeddings(None)


def test_infer_reconstruction(base_trainer):
    with pytest.raises(ValueError):
        base_trainer.infer_reconstruction(None)

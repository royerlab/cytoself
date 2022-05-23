import os.path
import tempfile
from shutil import rmtree
from unittest import TestCase
import torch

from ..basetrainer import BaseTrainer


class test_BaseTrainer(TestCase):
    def setUp(self):
        self._basepath = tempfile.mkdtemp()
        self.trainer = BaseTrainer({}, homepath=self._basepath)

    def test_init_(self):
        assert self.trainer.model is None
        assert self.trainer.best_model == []
        assert self.trainer.losses == {f'{p}_loss': [] for p in ['train', 'val', 'test']}
        assert self.trainer.lr == 0
        assert self.trainer.tb_writer is None
        assert self.trainer.optimizer is None
        assert self.trainer.savepath_dict['homepath'] == self._basepath
        assert self.trainer.current_epoch == 0

    def test__default_train_args(self):
        self.trainer._default_train_args()
        args = {
            'reducelr_patience': 4,
            'reducelr_increment': 0.1,
            'earlystop_patience': 12,
            'min_lr': 1e-8,
            'max_epochs': 100,
        }
        for key, val in args.items():
            assert self.trainer.train_args[key] == val

    def calc_loss_one_batch(self):
        assert self.trainer.calc_loss(torch.ones(5) * 3, torch.ones(5)) == torch.ones(1) * 4

    def test_set_optimizer(self):
        with self.assertRaises(ValueError):
            self.trainer.set_optimizer()

    def test_enable_tensorboard(self):
        self.trainer.savepath_dict['tb_logs'] = 'dummy'
        with self.assertWarns(UserWarning):
            self.trainer.enable_tensorboard()
        assert os.path.exists(self.trainer.savepath_dict['tb_logs'])

    def test_init_savepath(self):
        self.trainer.init_savepath()
        for key, val in self.trainer.savepath_dict.items():
            assert os.path.exists(val), key + ' path was not found.'

    def test_train_one_epoch(self):
        with self.assertRaises(ValueError):
            self.trainer.train_one_epoch(None)

    def test_calc_val_loss(self):
        with self.assertRaises(ValueError):
            self.trainer.calc_val_loss(None)

    def test__reduce_lr_on_plateau(self):
        with self.assertRaises(ValueError):
            self.trainer._reduce_lr_on_plateau(2)

    def test_fit(self):
        with self.assertRaises(ValueError):
            self.trainer.fit(None)

    def tearDown(self):
        rmtree(self._basepath)

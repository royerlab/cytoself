from os.path import join

import torch

from cytoself.datamanager.datamanager_oc import DataManagerOpenCell
from cytoself.datamanager.test.test_datamanager_oc import TmpDirTestCase
from cytoself.trainer.vanilla_trainer import VanillaAETrainer


class setup_VanillaAETrainer(TmpDirTestCase):
    def setUp(self):
        self.model_args = {
            'input_shape': (1, 32, 32),
            'emb_shape': (16, 16, 16),
            'output_shape': (1, 32, 32),
        }
        self._setup_model_args()
        self.trainer = VanillaAETrainer(self.model_args, self.train_args, homepath=self._basepath)
        self._setup_datamgr()

    def _setup_model_args(self):
        # Reduce model size to make test run fast
        self.model_args['encoder_args'] = {
            'in_channels': self.model_args['input_shape'][0],
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
            'out_channels': self.model_args['emb_shape'][0],
        }
        self.model_args['decoder_args'] = {
            'input_shape': self.model_args['emb_shape'],
            'num_residual_layers': 1,
            'output_shape': self.model_args['input_shape'],
        }
        self._setup_env()

    def _setup_env(self):
        super().setUp()
        self.gen_npy(self.model_args['input_shape'])
        self.train_args = {'lr': 1e-6, 'max_epochs': 2}

    def _setup_datamgr(self):
        self.datamgr = DataManagerOpenCell(self._basepath, ['nuc'], batch_size=2)
        self.datamgr.const_dataset()
        self.datamgr.const_dataloader()


class test_VanillaAETrainer(setup_VanillaAETrainer):
    def test_fit(self):
        self.trainer.fit(self.datamgr, tensorboard_path=join(self._basepath, 'tb_log'))
        assert len(self.trainer.losses['train_loss']) == self.train_args['max_epochs']
        assert min(self.trainer.losses['train_loss']) < torch.inf


class test_VanillaAETrainer_on_plateau(setup_VanillaAETrainer):
    def test__reduce_lr_on_plateau(self):
        self.train_args = {'lr': 1e-8, 'max_epochs': 5, 'reducelr_patience': 1, 'min_lr': 1e-9, 'earlystop_patience': 3}
        self.trainer = VanillaAETrainer(self.model_args, self.train_args, homepath=self._basepath)
        self.trainer.fit(self.datamgr, tensorboard_path=join(self._basepath, 'tb_log'))
        assert round(self.trainer.optimizer.param_groups[0]['lr'], 9) == 1e-9
        assert len(self.trainer.losses['train_loss']) == 4

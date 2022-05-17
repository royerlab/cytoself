from os.path import join
import torch

from cytoself.datamanager.datamanager_oc import DataManagerOpenCell
from ..vanilla import VanillaAE, VanillaAETrainer
from cytoself.datamanager.test.test_datamanager_oc import TmpDirTestCase


def test_VanillaAE():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape, emb_shape = (2, 100, 100), (64, 4, 4)
    model = VanillaAE(input_shape, emb_shape, input_shape)
    model.to(device)
    input_data = torch.randn((1,) + input_shape).to(device)
    out = model(input_data)
    assert out.shape == input_data.shape


class test_VanillaAETrainer(TmpDirTestCase):
    def setUp(self):
        self.model_args = {
            'input_shape': (1, 32, 32),
            'emb_shape': (16, 16, 16),
            'output_shape': (1, 32, 32),
        }
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
        super().setUp()
        self.gen_npy(self.model_args['input_shape'])
        self.train_args = {'lr': 1e-6, 'max_epochs': 2}
        self.trainer = VanillaAETrainer(self.model_args, self.train_args, homepath=self._basepath)
        self.datamgr = DataManagerOpenCell(self._basepath, ['nuc'], batch_size=2)
        self.datamgr.const_dataset()
        self.datamgr.const_dataloader()

    def test_fit(self):
        self.trainer.fit(self.datamgr, tensorboard_path=join(self._basepath, 'tb_log'))
        assert len(self.trainer.losses['train_loss']) == self.train_args['max_epochs']
        assert min(self.trainer.losses['train_loss']) < torch.inf

from os.path import join

import torch

from cytoself.trainer.test.test_vanilla_trainer import setup_VanillaAETrainer
from cytoself.trainer.cytoselflight_trainer import CytoselfLiteTrainer


class test_CytoselfLightTrainer(setup_VanillaAETrainer):
    def setUp(self):
        self.model_args = {
            'input_shape': (1, 32, 32),
            'emb_shapes': ((16, 16, 16), (16, 16, 16)),
            'output_shape': (1, 32, 32),
            'vq_args': {'num_embeddings': 7},
            'num_class': 3,
            'encoder_args': [
                {
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
                },
                {
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
                },
            ],
            'decoder_args': [{'num_residual_layers': 1}, {'num_residual_layers': 1}],
        }
        super()._setup_env()
        self.trainer = CytoselfLiteTrainer(self.model_args, self.train_args, homepath=self._basepath)
        self._setup_datamgr()

    def test_fit(self):
        self.trainer.fit(self.datamgr, tensorboard_path=join(self._basepath, 'tb_log'))
        assert len(self.trainer.losses['train_loss']) == self.train_args['max_epochs']
        assert min(self.trainer.losses['train_loss']) < torch.inf
        assert min(self.trainer.losses['train_vq_loss1']) < torch.inf
        assert min(self.trainer.losses['train_vq_loss2']) < torch.inf
        assert min(self.trainer.losses['train_fc_loss1']) < torch.inf
        assert min(self.trainer.losses['train_fc_loss2']) < torch.inf
        assert all([loss.requires_grad is False for loss in self.trainer.model.vq_loss])
        assert all([loss.requires_grad is False for loss in self.trainer.model.fc_loss])

    def test_infer_embeddings(self):
        with self.assertRaises(ValueError):
            self.trainer.infer_embeddings(None)

        out = self.trainer.infer_embeddings(self.datamgr.test_loader)
        assert out[0].shape == (len(self.datamgr.test_dataset),) + self.model_args['emb_shapes'][1]

        d = next(iter(self.datamgr.test_loader))
        out = self.trainer.infer_embeddings(d['image'].numpy())
        assert out.shape == (self.datamgr.batch_size,) + self.model_args['emb_shapes'][1]

    def test_infer_reconstruction(self):
        with self.assertRaises(ValueError):
            self.trainer.infer_embeddings(None)

        out = self.trainer.infer_reconstruction(self.datamgr.test_loader)
        assert out.shape == self.datamgr.test_dataset.data.shape

        d = next(iter(self.datamgr.test_loader))
        out = self.trainer.infer_reconstruction(d['image'].numpy())
        assert out.shape == (self.datamgr.batch_size,) + self.datamgr.test_dataset.data.shape[1:]

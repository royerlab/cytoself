import torch

from cytoself.trainer.test.test_vqvae_trainer import test_VQVAETrainer
from cytoself.trainer.vqvaefc_trainer import VQVAEFCTrainer


class test_VQVAEFCTrainer(test_VQVAETrainer):
    def setUp(self):
        self.model_args = {
            'input_shape': (1, 32, 32),
            'emb_shape': (16, 16, 16),
            'output_shape': (1, 32, 32),
            'vq_args': {'num_embeddings': 7},
            'num_class': 3,
        }
        self._setup_model_args()
        self.trainer = VQVAEFCTrainer(self.model_args, self.train_args, homepath=self._basepath)
        self._setup_datamgr()

    def test_fit(self):
        super().test_fit()
        assert min(self.trainer.losses['train_fc_loss']) < torch.inf

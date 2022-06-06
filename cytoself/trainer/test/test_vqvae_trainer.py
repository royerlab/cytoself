import torch

from cytoself.trainer.test.test_vanilla_trainer import test_VanillaAETrainer
from cytoself.trainer.vqvae_trainer import VQVAETrainer


class test_VQVAETrainer(test_VanillaAETrainer):
    def setUp(self):
        self.model_args = {
            'input_shape': (1, 32, 32),
            'emb_shape': (16, 16, 16),
            'output_shape': (1, 32, 32),
            'vq_args': {'num_embeddings': 7},
        }
        self._setup1()
        self.trainer = VQVAETrainer(self.model_args, self.train_args, homepath=self._basepath)
        self._setup_datamgr()

    def test_fit(self):
        super().test_fit()
        assert min(self.trainer.losses['train_vq_loss']) < torch.inf

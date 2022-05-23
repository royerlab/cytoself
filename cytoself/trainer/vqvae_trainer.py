from typing import Optional, Union, Collection

from torch import nn

from cytoself.trainer.autoencoder.vqvae import VQVAE
from cytoself.trainer.basetrainer import BaseTrainer


class VQVAETrainer(BaseTrainer):
    """
    Trainer object for VQ-VAE
    """

    def __init__(
        self,
        model_args: dict,
        train_args: dict,
        metrics_names: Collection[str] = ('loss', 'vq_loss', 'perplexity'),
        homepath: str = './',
        device: Optional[str] = None,
    ):
        super().__init__(train_args, metrics_names, homepath, device)
        self._init_model(VQVAE(**model_args))

    def calc_loss_one_batch(self, inputs, targets, vq_coeff: Union[int, float] = 1, **kwargs):
        """
        Computes loss

        Parameters
        ----------
        inputs : tensor
            input data
        targets : tensor
            target data
        vq_coeff : float
            coefficient for vq loss
        kwargs : dict
            kwargs for the loss function

        Returns
        -------
        tensor

        """
        reconstruction_loss = nn.MSELoss(**kwargs)(targets, inputs)
        loss = reconstruction_loss + vq_coeff * self.model.vq_loss
        return loss, self.model.vq_loss, self.model.perplexity

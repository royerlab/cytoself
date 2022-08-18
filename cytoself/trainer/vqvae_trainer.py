from typing import Optional

from torch import nn, Tensor

from cytoself.trainer.autoencoder.vqvae import VQVAE
from cytoself.trainer.basetrainer import BaseTrainer


class VQVAETrainer(BaseTrainer):
    """
    Trainer object for VQ-VAE
    """

    def __init__(self, model_args: dict, train_args: dict, homepath: str = './', device: Optional[str] = None):
        super().__init__(train_args, homepath, device)
        self._init_model(VQVAE(**model_args))

    def calc_loss_one_batch(
        self,
        inputs: Tensor,
        targets: Tensor,
        vq_coeff: float = 1,
        zero_grad: bool = False,
        backward: bool = False,
        optimize: bool = False,
        **kwargs,
    ) -> dict:
        """
        Computes loss

        Parameters
        ----------
        inputs : torch.Tensor
            input data
        targets : torch.Tensor
            target data
        vq_coeff : float
            coefficient for vq loss
        zero_grad : bool
            Sets the gradients of all optimized torch.Tensor s to zero.
        backward : bool
            Computes the gradient of current tensor w.r.t. graph leaves if True
        optimize : bool
            Performs a single optimization step if True
        kwargs : dict
            kwargs for the loss function

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        if zero_grad:
            self.optimizer.zero_grad()

        reconstruction_loss = nn.MSELoss(**kwargs)(targets, inputs)
        loss = reconstruction_loss + vq_coeff * self.model.vq_loss['loss']

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        output = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'perplexity': self.model.perplexity.item(),
        }
        output.update({'vq_' + k if k == 'loss' else k: v.item() for k, v in self.model.vq_loss.items()})

        return output

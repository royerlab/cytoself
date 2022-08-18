import inspect
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from cytoself.trainer.autoencoder.vqvaefc import VQVAEFC
from cytoself.trainer.basetrainer import BaseTrainer


class VQVAEFCTrainer(BaseTrainer):
    """
    Trainer object for VQ-VAE-FC
    """

    def __init__(self, model_args: dict, train_args: dict, homepath: str = './', device: Optional[str] = None):
        super().__init__(train_args, homepath, device)
        self._init_model(VQVAEFC(**model_args))

    def calc_loss_one_batch(
        self,
        inputs: Tensor,
        targets: Tensor,
        vq_coeff: float = 1,
        fc_coeff: float = 1,
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
            Input data
        targets : torch.Tensor
            Target data
        vq_coeff : float
            Coefficient for vq loss
        fc_coeff : float
            Coefficient for fc loss
        zero_grad : bool
            Sets the gradients of all optimized torch.Tensor s to zero.
        backward : bool
            Computes the gradient of current tensor w.r.t. graph leaves if True
        optimize : bool
            Performs a single optimization step if True
        kwargs : dict
            kwargs for loss functions

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        if zero_grad:
            self.optimizer.zero_grad()

        mse_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.MSELoss).args if a in kwargs}
        ce_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.CrossEntropyLoss).args if a in kwargs}
        reconstruction_loss = nn.MSELoss(**mse_kwargs)(inputs[0], targets[0])
        self.model.fc_loss = nn.CrossEntropyLoss(**ce_kwargs)(inputs[1], targets[1])
        loss = reconstruction_loss + vq_coeff * self.model.vq_loss['loss'] + fc_coeff * self.model.fc_loss

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        output = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'perplexity': self.model.perplexity.item(),
            'fc_loss': self.model.fc_loss.item(),
        }
        output.update({'vq_' + k if k == 'loss' else k: v.item() for k, v in self.model.vq_loss.items()})

        return output

    def train_one_epoch(self, data_loader: DataLoader, **kwargs) -> dict:
        """
        Trains self.model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.

        Returns
        -------
        Training metrics

        """
        _metrics = []
        for i, _batch in enumerate(data_loader):
            timg = self._get_data_by_name(_batch, 'image')
            tlab = self._get_data_by_name(_batch, 'label', force_float=False)
            loss = self.calc_loss_one_batch(
                self.model(timg), [timg, tlab], zero_grad=True, backward=True, optimize=True, **kwargs
            )

            # Accumulate metrics
            _metrics.append(loss)
        return self._aggregate_metrics(_metrics, 'train')

    @torch.inference_mode()
    def calc_val_loss(self, data_loader: DataLoader, **kwargs) -> dict:
        """
        Compute validate loss

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader for validation data

        Returns
        -------
        Validation loss

        """
        _metrics = []
        for i, _batch in enumerate(data_loader):
            vimg = self._get_data_by_name(_batch, 'image')
            vlab = self._get_data_by_name(_batch, 'label', force_float=False)
            _vloss = self.calc_loss_one_batch(self.model(vimg), [vimg, vlab])
            _metrics.append(_vloss)
        return self._aggregate_metrics(_metrics, 'val')

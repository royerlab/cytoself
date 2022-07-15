import inspect
from typing import Optional, Union, Collection

import torch
from torch import nn

from cytoself.trainer.autoencoder.vqvaefc import VQVAEFC
from cytoself.trainer.basetrainer import BaseTrainer


class VQVAEFCTrainer(BaseTrainer):
    """
    Trainer object for VQ-VAE-FC
    """

    def __init__(
        self,
        model_args: dict,
        train_args: dict,
        metrics_names: Collection[str] = ('loss', 'mse', 'vq_loss', 'perplexity', 'fc_loss'),
        homepath: str = './',
        device: Optional[str] = None,
    ):
        super().__init__(train_args, metrics_names, homepath, device)
        self._init_model(VQVAEFC(**model_args))

    def calc_loss_one_batch(
        self,
        inputs,
        targets,
        vq_coeff: Union[int, float] = 1,
        fc_coeff: Union[int, float] = 1,
        **kwargs,
    ):
        """
        Computes loss

        Parameters
        ----------
        inputs : tensor
            Input data
        targets : tensor
            Target data
        vq_coeff : float
            Coefficient for vq loss
        fc_coeff : float
            Coefficient for fc loss
        kwargs : dict
            kwargs for loss functions

        Returns
        -------
        tuple of tensors

        """
        mse_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.MSELoss).args if a in kwargs}
        ce_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.CrossEntropyLoss).args if a in kwargs}
        reconstruction_loss = nn.MSELoss(**mse_kwargs)(inputs[0], targets[0])
        self.model.fc_loss = nn.CrossEntropyLoss(**ce_kwargs)(inputs[1], targets[1])
        loss = reconstruction_loss + vq_coeff * self.model.vq_loss + fc_coeff * self.model.fc_loss
        return loss, reconstruction_loss, self.model.vq_loss, self.model.perplexity, self.model.fc_loss

    def train_one_epoch(self, data_loader, **kwargs):
        """
        Trains self.model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.

        Returns
        -------
        None

        """
        _metrics = [0] * len(self.metrics_names)
        for i, _batch in enumerate(data_loader):
            timg = self._get_data_by_name(_batch, 'image')
            tlab = self._get_data_by_name(_batch, 'label', force_float=False)
            self.optimizer.zero_grad()

            loss = self.calc_loss_one_batch(self.model(timg), [timg, tlab], **kwargs)
            loss[0].backward()

            # Adjust learning weights
            self.optimizer.step()

            # Accumulate metrics
            _metrics = [m + l.item() for m, l in zip(_metrics, loss)]
        _metrics = [m / i for m in _metrics]
        self.record_metrics(_metrics, phase='train')

    @torch.inference_mode()
    def calc_val_loss(self, data_loader, **kwargs):
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
        _metrics = [0] * len(self.metrics_names)
        for i, _batch in enumerate(data_loader):
            vimg = self._get_data_by_name(_batch, 'image')
            vlab = self._get_data_by_name(_batch, 'label', force_float=False)
            _vloss = self.calc_loss_one_batch(self.model(vimg), [vimg, vlab])
            _metrics = [m + l.item() for m, l in zip(_metrics, _vloss)]
        self.record_metrics(_metrics, phase='val')

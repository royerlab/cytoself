import inspect
from functools import partial
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cytoself.trainer.autoencoder.cytoselflite import CytoselfLite
from cytoself.trainer.basetrainer import BaseTrainer


class CytoselfLiteTrainer(BaseTrainer):
    """
    Trainer object for CytoselfLite model
    """

    def __init__(
        self,
        model_args: dict,
        train_args: dict,
        homepath: str = './',
        device: Optional[str] = None,
    ):
        metrics_names = ('loss', 'mse', 'vq_loss', 'perplexity', 'fc_loss')
        super().__init__(train_args, metrics_names, homepath, device)
        self._init_model(CytoselfLite(**model_args))

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
        mse_loss_fn, ce_loss_fn = nn.MSELoss(**mse_kwargs), nn.CrossEntropyLoss(**ce_kwargs)
        reconstruction_loss = mse_loss_fn(targets[0], inputs[0])
        self.model.fc_loss = [ce_loss_fn(t, i) for t, i in zip(targets[1:], inputs[1:])]
        loss = (
            reconstruction_loss
            + vq_coeff * torch.stack(self.model.vq_loss).sum()
            + fc_coeff * torch.stack(self.model.fc_loss).sum()
        )
        # TODO How to equalize losses?
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
        _metrics = [0, 0, [0, 0], [0, 0], [0, 0]]
        for i, _batch in enumerate(tqdm(data_loader, desc='Train')):
            timg = self._get_data_by_name(_batch, 'image')
            tlab = self._get_data_by_name(_batch, 'label')
            self.optimizer.zero_grad()

            loss = self.calc_loss_one_batch(self.model(timg), [timg, tlab, tlab], **kwargs)
            loss[0].backward()

            # Adjust learning weights
            self.optimizer.step()

            # Clear graphs to save memory
            self.model.fc_loss = self._detach_graph(self.model.fc_loss)
            self.model.vq_loss = self._detach_graph(self.model.vq_loss)

            # Accumulate metrics
            _metrics = [
                [_m + _l.item() for _m, _l in zip(m, l)] if isinstance(l, list) else m + l.item()
                for m, l in zip(_metrics, loss)
            ]
        _metrics = [[_m / i for _m in m] if isinstance(m, list) else m / i for m in _metrics]
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
        _metrics = [0, 0, [0, 0], [0, 0], [0, 0]]
        for i, _batch in enumerate(tqdm(data_loader, desc='Val  ')):
            vimg = self._get_data_by_name(_batch, 'image')
            vlab = self._get_data_by_name(_batch, 'label')
            _vloss = self.calc_loss_one_batch(self.model(vimg), [vimg, vlab, vlab])
            _metrics = [
                [_m + _l.item() for _m, _l in zip(m, l)] if isinstance(l, list) else m + l.item()
                for m, l in zip(_metrics, _vloss)
            ]
        self.record_metrics(_metrics, phase='val')

    @torch.inference_mode()
    def infer_embeddings(self, data, output_layer: str = 'vqvec1'):
        """
        Infers embeddings

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data
        output_layer : str
            Name & index of output layer

        Returns
        -------
        None

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self._infer_one_epoch(data, partial(self.model, output_layer=output_layer))
        else:
            return self.model(torch.from_numpy(data).float().to(self.device), output_layer).detach().cpu().numpy()

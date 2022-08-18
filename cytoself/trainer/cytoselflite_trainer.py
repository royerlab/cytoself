import inspect
from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
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
        """
        Initializes cytoself lite trainer

        Parameters
        ----------
        model_args : dict
            Arguments for model construction
        train_args : dict
            Arguments for training
        homepath : str
            Path where training results will be saved
        device : str
            Specify device; e.g. cpu, cuda, cuda:0 etc.
        """
        super().__init__(train_args, homepath, device)
        self._init_model(CytoselfLite(**model_args))

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
    ):
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
        tuple of tensors

        """
        if zero_grad:
            self.optimizer.zero_grad()

        mse_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.MSELoss).args if a in kwargs}
        ce_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(nn.CrossEntropyLoss).args if a in kwargs}
        mse_loss_fn, ce_loss_fn = nn.MSELoss(**mse_kwargs), nn.CrossEntropyLoss(**ce_kwargs)
        reconstruction_loss = mse_loss_fn(inputs[0], targets[0])
        self.model.fc_loss = {
            f'fc{j + 1}_loss': ce_loss_fn(t, i) for j, (t, i) in enumerate(zip(inputs[1:], targets[1:]))
        }
        loss = (
            reconstruction_loss
            + vq_coeff * torch.stack([d['loss'] for d in self.model.vq_loss.values()]).sum()
            + fc_coeff * torch.stack(list(self.model.fc_loss.values())).sum()
        )
        # TODO How to equalize losses?

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        output = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
        }
        output.update({k: v.item() for k, v in self.model.fc_loss.items()})
        output.update({k: v.item() for k, v in self.model.perplexity.items()})
        vq_loss_dict = {}
        for key0, val0 in self.model.vq_loss.items():
            for key1, val1 in val0.items():
                vq_loss_dict[key0 + '_' + key1] = val1.item()
        output.update(vq_loss_dict)

        return output

    def train_one_epoch(self, data_loader: DataLoader, **kwargs):
        """
        Trains self.model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.

        """
        _metrics = []
        for i, _batch in enumerate(tqdm(data_loader, desc='Train')):
            timg = self._get_data_by_name(_batch, 'image')
            tlab = self._get_data_by_name(_batch, 'label', force_float=False)
            loss = self.calc_loss_one_batch(
                self.model(timg), [timg, tlab, tlab], zero_grad=True, backward=True, optimize=True, **kwargs
            )

            # Accumulate metrics
            _metrics.append(loss)
        return self._aggregate_metrics(_metrics, 'train')

    @torch.inference_mode()
    def calc_val_loss(self, data_loader: DataLoader, **kwargs):
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
        for i, _batch in enumerate(tqdm(data_loader, desc='Val  ')):
            vimg = self._get_data_by_name(_batch, 'image')
            vlab = self._get_data_by_name(_batch, 'label', force_float=False)
            _vloss = self.calc_loss_one_batch(self.model(vimg), [vimg, vlab, vlab])
            _metrics.append(_vloss)
        return self._aggregate_metrics(_metrics, 'val')

    @torch.inference_mode()
    def infer_embeddings(self, data, output_layer: str = 'vqvec2'):
        """
        Infers embeddings

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data
        output_layer : str
            Name & index of output layer

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self._infer_one_epoch(data, partial(self.model, output_layer=output_layer))
        else:
            return self.model(torch.from_numpy(data).float().to(self.device), output_layer).detach().cpu().numpy()

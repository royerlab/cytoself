import inspect
from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from cytoself.trainer.autoencoder.cytoselffull import CytoselfFull
from cytoself.trainer.vqvae_trainer import VQVAETrainer


class CytoselfFullTrainer(VQVAETrainer):
    """
    Trainer object for CytoselfFull model
    """

    def __init__(
        self,
        train_args: dict,
        homepath: str = './',
        device: Optional[str] = None,
        model: Optional = None,
        model_args: Optional[dict] = None,
    ):
        """
        Initializes cytoself lite trainer

        Parameters
        ----------
        train_args : dict
            Arguments for training
        homepath : str
            Path where training results will be saved
        device : str
            Specify device; e.g. cpu, cuda, cuda:0 etc.
        model : Optional[torch.nn.Module]
            An autoencoder model class (uninitialized model)
        model_args : dict
            Arguments for model construction
        """
        if model is None:
            model = CytoselfFull
        super().__init__(train_args, homepath, device, model, model_args)

    def run_one_batch(
        self,
        batch: dict,
        variance: float,
        zero_grad: bool = False,
        backward: bool = False,
        optimize: bool = False,
        vq_coeff: float = 1,
        fc_coeff: float = 1,
        **kwargs,
    ) -> dict:
        """
        Computes loss for one batch

        Parameters
        ----------
        batch : dict
            A batch data from data loader
        variance : float
            Variance of the entire data. (required in VQVAE models)
        zero_grad : bool
            Sets the gradients of all optimized torch.Tensor s to zero.
        backward : bool
            Computes the gradient of current tensor w.r.t. graph leaves if True
        optimize : bool
            Performs a single optimization step if True
        vq_coeff : float
            Coefficient for vq loss
        fc_coeff : float
            Coefficient for fc loss
        kwargs : dict
            kwargs for loss functions

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        img = self.get_data_by_name(batch, 'image')
        lab = self.get_data_by_name(batch, 'label', force_float=False)
        model_outputs = self.model(img)

        if zero_grad:
            self.optimizer.zero_grad()

        loss = self._calc_losses(model_outputs, img, lab, variance, vq_coeff, fc_coeff)
        # TODO How to equalize losses?

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        output = {'loss': loss.item()}
        # Update additional losses
        self._update_loss_dict(output)

        return output

    def _calc_losses(
        self,
        model_outputs: tuple,
        img: Tensor,
        lab: Tensor,
        variance: float,
        vq_coeff: float,
        fc_coeff: float,
        **kwargs,
    ):
        """
        Calculate all losses

        Parameters
        ----------
        model_outputs : tuple
            Output from model
        img : torch.Tensor
            Image data from DataLoader
        lab : torch.Tensor
            Label data from DataLoader
        variance : float
            Data variance
        vq_coeff : float
            Coefficient for vq loss
        fc_coeff : float
            Coefficient for fc loss
        kwargs : dict
            kwargs for loss functions

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        mse_kwargs = {a: kwargs[a] for a in inspect.signature(nn.MSELoss).parameters if a in kwargs}
        ce_kwargs = {a: kwargs[a] for a in inspect.signature(nn.CrossEntropyLoss).parameters if a in kwargs}
        mse_loss_fn, ce_loss_fn = nn.MSELoss(**mse_kwargs), nn.CrossEntropyLoss(**ce_kwargs)
        self.model.mse_loss['reconstruction1_loss'] = mse_loss_fn(model_outputs[0], img) / variance
        # Matching the number of label target to the label output from the model.
        # The image output should always come to the first followed by label outputs.
        # self.model.fc_loss will be an empty dict when there is no label output from the model.
        self.model.fc_loss = {
            f'fc{self.model.fc_output_idx[j]}_loss': ce_loss_fn(t, i)
            for j, (t, i) in enumerate(zip(model_outputs[1:], [lab] * (len(model_outputs) - 1)))
        }
        return self._combine_losses(vq_coeff, fc_coeff)

    def _update_loss_dict(self, output: dict):
        """
        Update loss dict with additional component losses

        Parameters
        ----------
        output : dict
            Dict of losses

        """
        output.update({k: v.item() for k, v in self.model.fc_loss.items()})
        output.update({k: v.item() for k, v in self.model.perplexity.items()})
        output.update({k: v.item() for k, v in self.model.mse_loss.items()})
        vq_loss_dict = {}
        for key0, val0 in self.model.vq_loss.items():
            for key1, val1 in val0.items():
                vq_loss_dict[key0 + '_' + key1] = val1.item()
        output.update(vq_loss_dict)

    def _combine_losses(self, vq_coeff: float, fc_coeff: float):
        """
        Sum up all losses

        Parameters
        ----------
        vq_coeff : float
            Loss weight/coefficient for VQ losses
        fc_coeff : float
            Loss weight/coefficient for FC losses

        Returns
        -------
        torch.Tensor

        """
        fc_loss_list = list(self.model.fc_loss.values())
        vq_loss_list = [d['loss'] for d in self.model.vq_loss.values()]
        mse_loss_list = list(self.model.mse_loss.values())
        loss = (
            +(torch.stack(mse_loss_list).sum() if len(mse_loss_list) > 0 else 0)
            + vq_coeff * (torch.stack(vq_loss_list).sum() if len(vq_loss_list) > 0 else 0)
            + fc_coeff * (torch.stack(fc_loss_list).sum() if len(fc_loss_list) > 0 else 0)
        )
        return loss

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
            return self.infer_one_epoch(data, partial(self.model, output_layer=output_layer))
        else:
            return self.model(torch.from_numpy(data).float().to(self.device), output_layer).detach().cpu().numpy()

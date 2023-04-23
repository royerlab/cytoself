import inspect
from typing import Optional

from torch import Tensor, nn

from cytoself.trainer.autoencoder.vqvaefc import VQVAEFC
from cytoself.trainer.vqvae_trainer import VQVAETrainer


class VQVAEFCTrainer(VQVAETrainer):
    """
    Trainer object for VQ-VAE-FC
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
        Initializes VQVAEFC trainer

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
        model : Optional[torch.nn.Module]
            An autoencoder model class (uninitialized model)
        """
        if model is None:
            model = VQVAEFC
        super().__init__(train_args, homepath, device, model, model_args)

    # noinspection PyMethodOverriding
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
        Computes loss

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

        if zero_grad:
            self.optimizer.zero_grad()

        model_outputs = self.model(img)
        loss = self._calc_losses(model_outputs, img, lab, variance, vq_coeff, fc_coeff, **kwargs)

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
        self.model.mse_loss = nn.MSELoss(**mse_kwargs)(model_outputs[0], img) / variance
        self.model.fc_loss = nn.CrossEntropyLoss(**ce_kwargs)(model_outputs[1], lab)
        return self._combine_losses(vq_coeff, fc_coeff)

    def _update_loss_dict(self, output: dict):
        """
        Update loss dict with additional component losses

        Parameters
        ----------
        output : dict
            Dict of losses

        """
        output.update(
            {
                'reconstruction_loss': self.model.mse_loss.item(),
                'perplexity': self.model.perplexity.item(),
                'fc_loss': self.model.fc_loss.item(),
            }
        )
        output.update({'vq_' + k if k == 'loss' else k: v.item() for k, v in self.model.vq_loss.items()})

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
        return self.model.mse_loss + vq_coeff * self.model.vq_loss['loss'] + fc_coeff * self.model.fc_loss

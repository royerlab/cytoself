from typing import Optional

from torch import Tensor, nn
from tqdm import tqdm

from cytoself.trainer.autoencoder.vqvae import VQVAE
from cytoself.trainer.vanilla_trainer import VanillaAETrainer


class VQVAETrainer(VanillaAETrainer):
    """
    Trainer object for VQ-VAE
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
        Initializes VQVAE trainer

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
            model = VQVAE
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
            coefficient for vq loss
        kwargs : dict
            kwargs for the loss function

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        img = self.get_data_by_name(batch, 'image')

        if zero_grad:
            self.optimizer.zero_grad()

        model_outputs = self.model(img)
        loss = self._calc_losses(model_outputs, img, variance, vq_coeff, **kwargs)

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        output = {'loss': loss.item()}
        # Update additional losses
        self._update_loss_dict(output)

        return output

    def _calc_losses(self, model_outputs: tuple, img: Tensor, variance: float, vq_coeff: float, **kwargs):
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
        kwargs : dict
            kwargs for loss functions

        Returns
        -------
        A dict of tensors with the loss names and loss values.

        """
        self.model.mse_loss = nn.MSELoss(**kwargs)(model_outputs, img) / variance
        return self._combine_losses(vq_coeff)

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
            }
        )
        output.update({'vq_' + k if k == 'loss' else k: v.item() for k, v in self.model.vq_loss.items()})

    def _combine_losses(self, vq_coeff: float):
        """
        Sum up all losses

        Parameters
        ----------
        vq_coeff : float
            Loss weight/coefficient for VQ losses

        Returns
        -------
        torch.Tensor

        """
        return self.model.mse_loss + vq_coeff * self.model.vq_loss['loss']

    def run_one_epoch(self, datamanager, phase: str, **kwargs) -> dict:
        """
        Run one epoch of data on the model

        Parameters
        ----------
        datamanager : DataManager
            A DataManager object that has train, val and test data loader inside
        phase : str
            To indicate whether it's train, val or test phase

        Returns
        -------
        metrics in DataFrame

        """
        if self.model is None:
            raise ValueError('model is not defined.')
        else:
            is_train = phase.lower() == 'train'
            if phase == 'train':
                data_loader = datamanager.train_loader
                var = datamanager.train_variance
            elif phase == 'val':
                data_loader = datamanager.val_loader
                var = datamanager.val_variance
            elif phase == 'test':
                data_loader = datamanager.test_loader
                var = datamanager.test_variance
            else:
                raise ValueError('phase only accepts train, val or test.')
            _metrics = []
            for _batch in tqdm(data_loader, desc=f'{phase.capitalize():>5}'):
                loss = self.run_one_batch(
                    _batch, var, zero_grad=is_train, backward=is_train, optimize=is_train, **kwargs
                )
                _metrics.append(loss)
            return self._aggregate_metrics(_metrics, phase)

from typing import Optional

import torch

from cytoself.trainer.autoencoder.cytoselflite import CytoselfLite
from cytoself.trainer.cytoself_trainer_base import CytoselfTrainerBase


class CytoselfLiteTrainer(CytoselfTrainerBase):
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
        super().__init__(CytoselfLite(**model_args), train_args, homepath, device)

    def _update_loss_dict(self, output):
        output.update({k: v.item() for k, v in self.model.fc_loss.items()})
        output.update({k: v.item() for k, v in self.model.perplexity.items()})

    def _combine_losses(self, fc_coeff, reconstruction_loss, vq_coeff):
        loss = (
            reconstruction_loss
            + vq_coeff * torch.stack([d['loss'] for d in self.model.vq_loss.values()]).sum()
            + fc_coeff * torch.stack(list(self.model.fc_loss.values())).sum()
        )
        return loss

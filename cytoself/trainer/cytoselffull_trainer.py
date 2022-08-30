from typing import Optional

from cytoself.trainer.autoencoder.cytoselffull import CytoselfFull
from cytoself.trainer.cytoself_trainer_base import CytoselfTrainerBase


class CytoselfFullTrainer(CytoselfTrainerBase):
    """
    Trainer object for CytoselfFull model
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
        super().__init__(CytoselfFull(**model_args), train_args, homepath, device)

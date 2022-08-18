from typing import Optional

from cytoself.trainer.autoencoder.base import BaseAE
from cytoself.trainer.basetrainer import BaseTrainer


class VanillaAETrainer(BaseTrainer):
    """
    Trainer object for Vanilla autoencoder
    """

    def __init__(self, model_args: dict, train_args: dict, homepath: str = './', device: Optional[str] = None):
        """
        Initializes vanilla autoencoder trainer

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
        self._init_model(BaseAE(**model_args))

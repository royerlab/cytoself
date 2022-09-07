from typing import Optional

from cytoself.trainer.autoencoder.base import BaseAE
from cytoself.trainer.basetrainer import BaseTrainer


class VanillaAETrainer(BaseTrainer):
    """
    Trainer object for Vanilla autoencoder
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
        Initializes vanilla autoencoder trainer

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
            model = BaseAE
        if model_args is None:
            raise ValueError('model_args is required when model is None.')
        else:
            model = model(**model_args)
        super().__init__(train_args, homepath, device, model)

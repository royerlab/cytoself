from typing import Optional

from cytoself.trainer.autoencoder.base import BaseAE
from cytoself.trainer.basetrainer import BaseTrainer


class VanillaAETrainer(BaseTrainer):
    """
    Trainer object for Vanilla autoencoder
    """

    def __init__(self, model_args: dict, train_args: dict, homepath: str = './', device: Optional[str] = None):
        super().__init__(train_args, homepath, device)
        self._init_model(BaseAE(**model_args))

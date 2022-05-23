from typing import Optional, Collection

from cytoself.trainer.autoencoder.vanilla import VanillaAE
from cytoself.trainer.basetrainer import BaseTrainer


class VanillaAETrainer(BaseTrainer):
    """
    Trainer object for Vanilla autoencoder
    """

    def __init__(
        self,
        model_args: dict,
        train_args: dict,
        metrics_names: Collection[str] = ('loss',),
        homepath: str = './',
        device: Optional[str] = None,
    ):
        super().__init__(train_args, metrics_names, homepath, device)
        self._init_model(VanillaAE(**model_args))

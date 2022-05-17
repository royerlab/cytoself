import inspect
import os
from os.path import join
from typing import Optional
from warnings import warn

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    """
    Base class for Trainer
    """

    def __init__(
        self,
        homepath: str = './',
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.train_args = {}
        self.model = None
        self.best_model = []
        self.losses = None
        self.lr = 0
        self.tb_writer = None
        self.tb_savepath = None
        self.optimizer = None
        self.savepath_dict = {'homepath': homepath}
        self.current_epoch = 0

    def _default_train_args(self, args: Optional[dict] = None):
        """
        Sets default training arguments

        Parameters
        ----------
        args : dict
            arguments

        Returns
        -------
        None

        """
        if args is None:
            args = {
                'reducelr_patience': 4,
                'reducelr_increment': 0.1,
                'earlystop_patience': 12,
                'min_lr': 1e-8,
                'max_epochs': 100,
            }
        for key, val in args.items():
            if key not in self.train_args:
                self.train_args[key] = val

    def calc_loss(self, inputs, targets, loss_fn: Optional = None, **kwargs):
        """
        Computes loss

        Parameters
        ----------
        inputs : tensor
            input data
        targets : tensor
            target data
        loss_fn : pytorch loss function
            loss function
        kwargs : dict
            kwargs for the loss function

        Returns
        -------
        tensor

        """
        if loss_fn is None:
            loss_fn = nn.MSELoss
        return loss_fn(**kwargs)(inputs, targets)

    def set_optimizer(self, optimizer: Optional = None, **kwargs):
        """
        Sets optimizer

        Parameters
        ----------
        optimizer : pytorch optimizer
            optimizer

        Returns
        -------
        None

        """
        if self.model:
            local_optimizer = optim.Adam if optimizer is None else optimizer
            local_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(local_optimizer).args if a in kwargs}
            self.optimizer = local_optimizer(self.model.parameters(), **local_kwargs)
        else:
            raise ValueError("self.model attribute is not initialized...")

    def enable_tensorboard(self, savepath='tb_logs', **kwargs):
        """
        Enables tensorboad

        Parameters
        ----------
        savepath : str
            save path for tensorboard log

        Returns
        -------
        None

        """
        if 'tb_logs' in self.savepath_dict:
            warn(
                f'TensorBoard save path has been changed from {self.savepath_dict["tb_logs"]} to {savepath}',
                UserWarning,
            )
        self.savepath_dict['tb_logs'] = savepath
        if not os.path.exists(self.savepath_dict['tb_logs']):
            os.makedirs(self.savepath_dict['tb_logs'])
        self.tb_writer = SummaryWriter(self.savepath_dict['tb_logs'])

    def init_savepath(self, makedirs: bool = True, **kwargs):
        """
        Initializes saving folders

        Parameters
        ----------
        makedirs : bool
            make directories if True

        Returns
        -------
        None

        """
        directories = ['checkpoints', 'embeddings', 'ft_analysis', 'umaps', 'visualization']
        for d in directories:
            self.savepath_dict[d] = join(self.savepath_dict['homepath'], d)
            if makedirs and not os.path.exists(self.savepath_dict[d]):
                os.makedirs(self.savepath_dict[d])

    def train_one_epoch(self, **kwargs):
        raise NotImplementedError

    def calc_val_loss(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

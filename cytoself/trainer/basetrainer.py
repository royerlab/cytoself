import inspect
import os
from copy import deepcopy
from os.path import join
from typing import Optional, Union
from warnings import warn

from natsort import natsorted
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class BaseTrainer:
    """
    Base class for Trainer
    """

    def __init__(
        self, train_args: dict, homepath: str = './', device: Optional[str] = None, model: Optional = None, **kwargs
    ):
        """
        Base class for trainer

        Parameters
        ----------
        train_args : dict
            Training arguments
        homepath : str
            Path where training results will be saved
        device : str
            Specify device; e.g. cpu, cuda, cuda:0 etc.
        model : Optional[torch.nn.Module] instance
            An autoencoder model instance
        """
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.train_args = train_args
        self.model = None
        self.best_model = []
        self.lr = 0
        self.tb_writer = None
        self.optimizer = None
        self.savepath_dict = {'homepath': homepath}
        self.current_epoch = 1
        self.history = pd.DataFrame()
        if model is not None:
            self._init_model(model)

    def _init_model(self, model):
        """
        Initializes model

        Parameters
        ----------
        model : model object
            The autoencoder model

        """
        self.model = model
        self.model.to(self.device)
        # optimizer should be set after model moved to other devices
        self._default_train_args()
        self.set_optimizer(**self.train_args)
        self.init_savepath()

    def _default_train_args(self):
        """
        Sets default training arguments to make sure the model trainer has at least the following arguments.
        """
        args = {
            'reducelr_patience': 4,
            'reducelr_increment': 0.1,
            'earlystop_patience': 12,
            'min_lr': 1e-8,
            'max_epoch': 100,
        }
        for key, val in args.items():
            if key not in self.train_args:
                self.train_args[key] = val

    def run_one_batch(
        self,
        batch: dict,
        *args,
        zero_grad: bool = False,
        backward: bool = False,
        optimize: bool = False,
        **kwargs,
    ) -> dict:
        """
        Computes loss for one batch

        Parameters
        ----------
        batch : dict
            A batch data from data loader
        zero_grad : bool
            Sets the gradients of all optimized torch.Tensor s to zero.
        backward : bool
            Computes the gradient of current tensor w.r.t. graph leaves if True
        optimize : bool
            Performs a single optimization step if True
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
        loss = self._calc_losses(model_outputs, img)

        if backward:
            loss.backward()

        if optimize:
            self.optimizer.step()

        return {'loss': loss.item()}

    def _calc_losses(self, model_outputs, img, *args, **kwargs):
        return nn.MSELoss(**kwargs)(model_outputs, img)

    def _aggregate_metrics(self, metrics: dict, phase: str) -> dict:
        """
        Aggregate a list of dicts of metrics in a epoch

        Parameters
        ----------
        metrics : dict
            A list of metrics in dict
        phase : str
            train or val or test; will be on the top of the column names

        Returns
        -------
        A DataFrame with all metrics in it.

        """
        metrics = pd.DataFrame(metrics).mean(axis=0).to_frame().T
        metrics.columns = [phase + '_' + c for c in metrics.columns]
        return metrics

    def record_metrics(self, metrics: Union[list[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """
        Register metrics to self.history

        Parameters
        ----------
        metrics : (a list of) DataFrame
            A dataframe of metric values

        """

        if isinstance(metrics, list) or isinstance(metrics, tuple):
            metrics = pd.concat(metrics, axis=1)
        self.history = pd.concat([self.history, metrics], ignore_index=True, axis=0)
        self.history = self.history.fillna(0)

    def set_optimizer(self, optimizer: Union[str, optim.Optimizer] = 'AdamW', **kwargs):
        """
        Sets optimizer

        Parameters
        ----------
        optimizer : torch.optim.Optimizer or str
            Optimizer name or Optimizer object

        """
        if self.model:
            if isinstance(optimizer, str):
                if optimizer == 'Adam':
                    local_optimizer = optim.Adam
                elif optimizer == 'AdamW':
                    local_optimizer = optim.AdamW
                else:
                    raise ValueError(
                        optimizer
                        + ' cannot be specified with string. Please pass a torch.optim.Optimizer object directly'
                    )
            else:
                local_optimizer = optimizer
            local_kwargs = {a: kwargs[a] for a in inspect.signature(local_optimizer).parameters if a in kwargs}
            self.optimizer = local_optimizer(self.model.parameters(), **local_kwargs)
        else:
            raise ValueError("self.model attribute is not initialized...")

    def enable_tensorboard(self, savepath='tb_logs', **kwargs):
        """
        Enables TensorBoard

        Parameters
        ----------
        savepath : str
            save path for tensorboard log

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

    def write_on_tensorboard(self, tensorboard_path: str):
        """
        Write training history to tensorboard

        Parameters
        ----------
        tensorboard_path : str
            The path to save tensorboard log

        """
        if tensorboard_path is not None:
            if self.tb_writer is None:
                self.enable_tensorboard(tensorboard_path)
            m_names = (
                self.history.columns.to_frame()
                .iloc[:, 0]
                .str.replace('train_', '')
                .str.replace('val_', '')
                .str.replace('test_', '')
                .to_frame()
                .iloc[:, 0]
                .str.split('_', expand=True)
                .iloc[:, 0]
                .unique()
            )
            for tag in m_names:
                if tag == 'loss':
                    df = self.history.filter(items=[f'{i}_loss' for i in ['train', 'val', 'test']], axis=1)
                else:
                    df = self.history.filter(regex=tag.lower(), axis=1)
                self.tb_writer.add_scalars(tag, df.to_dict('index')[len(self.history) - 1], len(self.history))
            self.tb_writer.flush()

    def init_savepath(self, makedirs: bool = True, **kwargs):
        """
        Initializes saving folders

        Parameters
        ----------
        makedirs : bool
            make directories if True

        """
        directories = ['checkpoints', 'embeddings', 'visualization']
        for d in directories:
            self.savepath_dict[d] = join(self.savepath_dict['homepath'], d)
            if makedirs and not os.path.exists(self.savepath_dict[d]):
                os.makedirs(self.savepath_dict[d])

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
            elif phase == 'val':
                data_loader = datamanager.val_loader
            elif phase == 'test':
                data_loader = datamanager.test_loader
            else:
                raise ValueError('phase only accepts train, val or test.')
            _metrics = []
            for _batch in tqdm(data_loader, desc=f'{phase.capitalize():>5}'):
                loss = self.run_one_batch(_batch, zero_grad=is_train, backward=is_train, optimize=is_train, **kwargs)
                _metrics.append(loss)
            return self._aggregate_metrics(_metrics, phase)

    @torch.inference_mode()
    def infer_one_epoch(self, data_loader: DataLoader, _model):
        """
        Infers the output of a given model for one epoch

        Parameters
        ----------
        data_loader : DataLoader
            A DataLoader object that handles data distribution and augmentation.
        _model : model
            A model object
        Returns
        -------
        Numpy array

        """
        output, output_label = [], []
        for i, _batch in enumerate(tqdm(data_loader, desc='Infer')):
            timg = self.get_data_by_name(_batch, 'image')
            out = _model(timg)
            if not torch.is_tensor(out):
                out = out[0]
            output.append(out.detach().cpu().numpy())
            if 'label' in _batch:
                output_label.append(_batch['label'])
        output = np.vstack(output)
        if len(output_label) == len(output):
            output_label = np.vstack(output_label)
        else:
            _output_label = np.hstack(output_label)
            if len(_output_label) == len(output):
                output_label = _output_label
            else:
                output_label = np.array([])
        return output, output_label

    def get_data_by_name(self, data: dict, name: str, force_float=True) -> Tensor:
        """
        Get tensor by name when the output of dataloader is dict.

        Parameters
        ----------
        data : dict
            Dictionary of tensor
        name : str
            Key of dict
        force_float : bool
            Force the output to be float if True

        Returns
        -------
        Tensor

        """
        output = data[name]
        if force_float:
            output = output.float()
        return output.to(self.device)

    def _reduce_lr_on_plateau(self, count_lr_no_improve: int) -> int:
        """
        Reduces learning rate when no improvement in the training

        Parameters
        ----------
        count_lr_no_improve : int
            Number of epochs with no improvement

        Returns
        -------
        int

        """
        if self.optimizer is None:
            raise ValueError('optimizer is not defined.')
        else:
            if count_lr_no_improve >= self.train_args['reducelr_patience']:
                if self.optimizer.param_groups[0]['lr'] > self.train_args['min_lr']:
                    self.optimizer.param_groups[0]['lr'] *= self.train_args['reducelr_increment']
                    print('learn rate = ', self.optimizer.param_groups[0]['lr'])
                    return 0
                else:
                    return count_lr_no_improve
            else:
                return count_lr_no_improve

    def fit(
        self,
        datamanager,
        initial_epoch: int = 1,
        tensorboard_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Fit pytorch model

        Parameters
        ----------
        datamanager : DataManager
            DataManager object
        initial_epoch : int
            Epoch at which to start training (useful for resuming a previous training run).
        tensorboard_path : str
            Path for Tensorboard to load logs

        """
        stop = False
        if self.model is None:
            raise ValueError('model is not defined.')
        else:
            self.current_epoch = initial_epoch
            best_vloss = torch.inf if 'val_loss' not in self.history else min(self.history['val_loss'])
            count_lr_no_improve = 0
            count_early_stop = 0
            for current_epoch in range(self.current_epoch, self.train_args['max_epoch'] + 1):
                if stop:
                    break
                else:
                    self.current_epoch = current_epoch
                    print(f'Epoch {current_epoch}/{self.train_args["max_epoch"]}')
                    # Train the model
                    self.model.train(True)
                    train_metrics = self.run_one_epoch(datamanager, 'train', **kwargs)
                    self.model.train(False)

                    # Validate the model
                    with torch.inference_mode():
                        val_metrics = self.run_one_epoch(datamanager, 'val', **kwargs)

                    # Register learning rate
                    lr_metrics = pd.DataFrame({'lr': [self.optimizer.param_groups[0]['lr']]})
                    metrics_all = [train_metrics, val_metrics, lr_metrics]

                    # Track the best performance, and save the model's state
                    _vloss = np.nan_to_num(val_metrics['val_loss'].iloc[-1])
                    if _vloss < best_vloss:
                        best_vloss = _vloss
                        self.best_model = deepcopy(self.model)
                        # Save the best model checkpoint
                        self.save_checkpoint()
                    else:
                        count_lr_no_improve += 1
                        count_early_stop += 1

                    # Reduce learn rate on plateau
                    count_lr_no_improve = self._reduce_lr_on_plateau(count_lr_no_improve)

                    # Check for early stopping
                    if count_early_stop >= self.train_args['earlystop_patience']:
                        print('Early stopping.')
                        stop = True

                    if stop or current_epoch == self.train_args["max_epoch"]:
                        # Test the model with test data
                        with torch.inference_mode():
                            test_metrics = self.run_one_epoch(datamanager, 'test', **kwargs)
                        metrics_all.append(test_metrics)

                    # Record metrics
                    self.record_metrics(metrics_all)

                    # Record logs for TensorBoard
                    if tensorboard_path is not None:
                        tensorboard_path = join(self.savepath_dict['homepath'], tensorboard_path)
                    self.write_on_tensorboard(tensorboard_path)

            self.save_model(self.savepath_dict['homepath'], f'model_{self.current_epoch}.pt')
            self.history.to_csv(join(self.savepath_dict['visualization'], 'training_history.csv'), index=False)

    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save a model checkpoint

        Parameters
        ----------
        path : str
            Path to save model checkpoints

        """
        if path is None:
            path = self.savepath_dict['checkpoints']
        fpath = join(path, f'checkpoint_ep{self.current_epoch}.chkp')
        torch.save(
            {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
            },
            fpath,
        )
        print('A model checkpoint has been saved at ' + fpath)

    def load_checkpoint(self, path: Optional[str] = None, epoch: Optional[int] = None):
        if path is None:
            path = self.savepath_dict['checkpoints']
        if epoch is None:
            fpath = join(path, natsorted([f for f in os.listdir(path) if f.endswith('.chkp')])[-1])
        else:
            fpath = join(path, f'checkpoint_ep{epoch}.chkp')

        checkpoint = torch.load(fpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        print(fpath + ' has been loaded.')

    def save_model(
        self, path: str, filename: str = 'pytorch_model.pt', model: Optional[nn.Module] = None, by_weights: bool = False
    ):
        """
        Save a pytorch model

        Parameters
        ----------
        path : str
            Path to save the model
        filename : str
            File name
        model : nn.Module
            Pytorch model
        by_weights : bool
            Only save weights in dict if True.

        """
        if model is None:
            model = self.model
        fpath = join(path, filename)
        torch.save(model.state_dict() if by_weights else model, fpath)
        print('A model has been saved at ' + fpath)

    def load_model(self, path: str, by_weights: bool = True):
        """
        Load a pytorch model

        Parameters
        ----------
        path : str
            Path to the pytorch model
        by_weights : bool
            Load model by weights if True.
            Loading by weights is safer in case some logic has been changed from when the model was saved.

        """
        _model = torch.load(path)
        if isinstance(_model, dict):
            self.model.load_state_dict(_model)
        else:
            if by_weights:
                self.model.load_state_dict(_model.state_dict())
            else:
                self.model = _model
        print(f'A model has been loaded from {path}')

    @torch.inference_mode()
    def infer_embeddings(self, data):
        """
        Infers embeddings

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self.infer_one_epoch(data, self.model.encoder)
        else:
            return self.model.encoder(torch.from_numpy(data).float().to(self.device)).detach().cpu().numpy()

    @torch.inference_mode()
    def infer_reconstruction(self, data):
        """
        Infers decoded images

        Parameters
        ----------
        data : numpy array or DataLoader
            Image data

        """
        if data is None:
            raise ValueError('The input to infer_embeddings cannot be None.')
        if isinstance(data, DataLoader):
            return self.infer_one_epoch(data, self.model)[0]
        else:
            output = self.model(torch.from_numpy(data).float().to(self.device))
            if isinstance(output, tuple) or isinstance(output, list):
                return output[0].detach().cpu().numpy()
            else:
                return output.detach().cpu().numpy()

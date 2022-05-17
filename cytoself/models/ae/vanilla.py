from os.path import join
from tqdm import tqdm
from typing import Optional
import torch
from torch import nn, Tensor

from cytoself.models.encoders.efficientenc2d import efficientenc_b0
from cytoself.models.decoders.resnet2d import DecoderResnet
from cytoself.models.trainer.basetrainer import BaseTrainer


class VanillaAE(nn.Module):
    """
    Vanilla Autoencoder model
    """

    def __init__(
        self,
        input_shape: tuple,
        emb_shape: tuple,
        output_shape: tuple,
        encoder_args: Optional[dict] = None,
        decoder_args: Optional[dict] = None,
        encoder: Optional = None,
        decoder: Optional = None,
    ):
        super().__init__()
        if encoder is None:
            encoder = efficientenc_b0
        if decoder is None:
            decoder = DecoderResnet
        if encoder_args is None:
            encoder_args = {'in_channels': input_shape[0], 'out_channels': emb_shape[0]}
        if decoder_args is None:
            decoder_args = {'input_shape': emb_shape, 'output_shape': output_shape}
        self.encoder = encoder(**encoder_args)
        self.decoder = decoder(**decoder_args)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VanillaAETrainer(BaseTrainer):
    """
    A comprehensive Trainer object
    """

    def __init__(
        self,
        model_args: dict,
        train_args: dict,
        homepath: str = './',
        device: Optional[str] = None,
    ):
        super().__init__(homepath, device)
        self.train_args = train_args
        self.losses = {'train_loss': [], 'val_loss': [], 'test_loss': []}
        self.model = VanillaAE(**model_args)
        self.model.to(self.device)
        # optimizer should be set after model moved to other devices
        self.set_optimizer(**train_args)
        self._default_train_args()

    def _default_train_args(self):
        super()._default_train_args()

    def train_one_epoch(self, data_loader):
        """
        Trains model for one epoch

        Parameters
        ----------
        data_loader: data loader

        Returns
        -------
        Mean training loss for the epoch

        """
        _tloss = 0.0
        for i, tdata in enumerate(data_loader):
            timg = tdata['image'].float().to(self.device)
            self.optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = self.calc_loss(self.model(timg), timg)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Accumulate losses
            _tloss += loss.item()
        return _tloss / (i + 1)

    def calc_val_loss(self, data_loader):
        """
        Compute validate loss

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader for validation data

        Returns
        -------
        Validation loss

        """
        _vloss = 0.0
        for i, vdata in enumerate(data_loader):
            vimg = vdata['image'].float().to(self.device)
            _vloss += self.calc_loss(self.model(vimg), vimg).item()
        return _vloss / (i + 1)

    def fit(
        self,
        datamanager,
        initial_epoch: int = 0,
        tensorboard_path: Optional[str] = None,
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

        Returns
        -------
        None

        """
        self.current_epoch = initial_epoch
        max_epochs = self.train_args['max_epochs']
        reducelr_patience = self.train_args['reducelr_patience']
        reducelr_increment = self.train_args['reducelr_increment']
        min_lr = self.train_args['min_lr']
        earlystop_patience = self.train_args['earlystop_patience']

        best_vloss = torch.inf if len(self.losses['val_loss']) == 0 else min(self.losses['val_loss'])
        count_lr_no_improve = 0
        count_early_stop = 0
        for _ in tqdm(range(self.current_epoch, max_epochs)):

            # Train the model
            self.model.train(True)
            self.losses['train_loss'].append(self.train_one_epoch(datamanager.train_loader))
            self.model.train(False)

            # Validate the model
            _vloss = self.calc_val_loss(datamanager.val_loader)
            self.losses['val_loss'].append(_vloss)

            # Track the best performance, and save the model's state
            if _vloss < best_vloss:
                best_vloss = _vloss
                self.best_model = self.model.state_dict()
            else:
                count_lr_no_improve += 1
                count_early_stop += 1

            # Reduce learn rate on plateau
            if count_lr_no_improve >= reducelr_patience:
                if self.optimizer.param_groups[0]['lr'] > min_lr:
                    self.optimizer.param_groups[0]['lr'] *= reducelr_increment
                    print('learn rate = ', self.optimizer.param_groups[0]['lr'])
                    count_lr_no_improve = 0

            # Record logs for TensorBoard
            if tensorboard_path is not None:
                if self.tb_writer is None:
                    self.enable_tensorboard(tensorboard_path)
                self.tb_writer.add_scalars(
                    'Loss', {'Training': self.losses['train_loss'][-1], 'Validation': _vloss}, self.current_epoch + 1
                )
                self.tb_writer.flush()
            self.current_epoch += 1

            # Check for early stopping
            if count_early_stop >= earlystop_patience:
                break

        torch.save(self.best_model, join(self.savepath_dict['homepath'], f'model_{self.current_epoch + 1}.pt'))

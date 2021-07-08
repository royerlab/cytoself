import os
import traceback
from collections.abc import Iterable

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.keras.layers import Flatten, Reshape
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K

from cytoself.models.splitq import CytoselfSQModel
from cytoself.external.efficientnet2d import EfficientEncoderB0
from cytoself.external.layers.vq import VectorQuantizer
from cytoself.components.blocks.fc_block import fc_block
from cytoself.components.decoders.resnet import Decoderres_bicub
from cytoself.components.layers.norm_mse import normalized_mse
from cytoself.components.layers.loss import SimpleMSE
from cytoself.components.layers.upconcat import UpConcat2D
from cytoself.components.layers.histcounts import Histcounts
from cytoself.components.metrics.metric import Metric

from cytoself.analysis.plot_functions.plot_reconstruction_panel import (
    plot_reconstruction_panel,
)
from cytoself.analysis.plot_functions.plot_histories import plothistory
from cytoself.utils.path_utils import split_path_all


class CytoselfFullModel(CytoselfSQModel):
    """
    A class object for the full EfficientNet-ResNet VQVAE model which has split quantization and pretext identification.
    """

    def __init__(
        self,
        input_image_shape,
        output_image_shape=None,
        q_splits=[1, 9],
        output_dir="model_outputs",
        suffix_dirs="",
        encoder_block_args=None,
        learn_rate=1e-4,
        early_stop_lr=1e-8,
        callbacks_start_from=1,
        num_embeddings=(2048, 2048),
        embedding_dims=(64, 64),
        commitment_costs=(0.25, 0.25),
        num_fc_units=1000,
        num_fc_output_classes=1363,
        num_fc_layers=2,
        fc_layer_connections=[1, 2],
        num_residual_layers=2,
        activation="swish",
        early_stop_patience=12,
        reduce_lr_patience=4,
        loss_weights={"decoder": 1, "fc": 1, "VQ": 1},
        use_depthwise=False,
        **kwargs,
    ):
        """
        :param input_image_shape: dimensions of input image
        :param output_image_shape:  dimensions of output image
        :param q_splits: a list of number of splits in each VQ layer
        :param output_dir: base path for saving output files
        :param suffix_dirs: suffix for saving directory (custom comments on the directory)
        :param encoder_block_args: dict of params for encoder (EfficientNet)
        :param learn_rate: learn rate
        :param early_stop_lr: early stop learn rate
        :param callbacks_start_from: epoch number where callbacks start monitoring
        :param num_embeddings: number of codes in the codebooks in each VQ layer
        :param embedding_dims: dimensions of embeddings in each VQ layer
        :param commitment_costs: commitment cost in each VQ layer
        :param num_fc_units: number of units in each fc layer of the fc block
        :param num_fc_output_classes: number of output classes in the fc block
        :param num_fc_layers: number of layers in the fc block
        :param fc_layer_connections: list of vq layers that connect to fc layer (e.g. 1: vq1; 0 is after concat of all vq)
        :param num_residual_layers: number of residual layers in decoders
        :param activation: activation function in all activation layers
        :param early_stop_patience: early stop patience
        :param reduce_lr_patience: Reduce learn rate patience
        :param loss_weights: a dict of coeff. to balance loss weights of multiple model outputs.
        :param use_depthwise: use depthwise conv in decoder if True
        :param dropout_rate: dropout ratio
        """

        kwargs.setdefault("dropout_rate", 0.5)

        # set the designated vqvec all zeros
        # this is to reconstruct images with all zeros in the designated vqvec
        # so that the reconstruction will fully based on the other layer(s).
        kwargs.setdefault("zero_vqvec_idx", [])

        if output_image_shape is None:
            output_image_shape = input_image_shape

        self.num_fc_units = num_fc_units
        self.num_fc_output_classes = num_fc_output_classes
        self.num_fc_layers = num_fc_layers
        self.fc_layer_connections = fc_layer_connections
        self.fc_blocks = []
        self.fc_outputs = []
        self.dropout_rate = kwargs["dropout_rate"]

        super().__init__(
            input_image_shape,
            output_image_shape,
            q_splits,
            output_dir,
            suffix_dirs=suffix_dirs,
            encoder_block_args=encoder_block_args,
            learn_rate=learn_rate,
            early_stop_lr=early_stop_lr,
            callbacks_start_from=callbacks_start_from,
            num_embeddings=num_embeddings,
            embedding_dims=embedding_dims,
            commitment_costs=commitment_costs,
            num_residual_layers=num_residual_layers,
            activation=activation,
            early_stop_patience=early_stop_patience,
            reduce_lr_patience=reduce_lr_patience,
            loss_weights=loss_weights,
            use_depthwise=use_depthwise,
            **kwargs,
        )

    def init_savepath(self, path_dict=None, domakedirs=True):
        """
        Create directories for saving outcomes.
        :param path_dict: home path for the output path
        :param domakedirs: make directories if True
        """
        if self.kwargs.get("init_savepath"):
            # One can create custom structured folder hierarchy here.
            self.savepath_dict["metadata"] = ""
            self.mkdirs_savepath_dict(path_dict=path_dict, makedirs=domakedirs)

    def construct_model(self):
        """
        Construct a VQVAE model
        """
        self.encoders = []
        self.decoders = []
        self.pre_vq = []
        self.post_vq = []
        self.vq_lyrs = []
        self.mse_lyrs = []
        self.histcounters = []
        self.upconcat_list = []
        for i in range(self.num_encs):
            # encoders are built from outerside to inside
            self.encoders += [
                EfficientEncoderB0(
                    input_shape=self.input_image_shape
                    if i == 0
                    else self.post_vq_shape[i - 1],
                    name=f"encoder{i + 1}",
                    include_top=False,
                    blocks_args=self.encoder_block_args[i],
                    isencoder=1 if i == 0 else 2,
                    num_hiddens_last=self.embedding_dims[i] * self.q_splits[i],
                )
            ]
            # decoders are built from outerside to inside
            self.decoders += [
                Decoderres_bicub(
                    self.post_vq_shape[i]
                    if i == self.num_encs - 1
                    else self.post_vq_shape[i][:-1]
                    + [sum([j[-1] for j in self.post_vq_shape[i:]])],
                    self.output_image_shape if i == 0 else self.post_vq_shape[i - 1],
                    num_hiddens=self.embedding_dims[i],
                    num_residual_layers=self.num_residual_layers,
                    num_hidden_decrease=True,
                    min_hiddens=32,
                    act=self.activation,
                    include_last=True if i == 0 else False,
                    use_depthwise=self.use_depthwise,
                    name=f"decoder{i + 1}",
                )
            ]

            self.pre_vq += [Reshape(self.pre_vq_shape[i], name=f"pre_vq{i + 1}")]

            self.vq_lyrs += [
                VectorQuantizer(
                    embedding_dim=self.embedding_dims[i],
                    num_embeddings=self.num_embeddings[i],
                    commitment_cost=self.commitment_costs[i],
                    n_outputs=2,
                    coeff=self.loss_weights["VQ"][i]
                    if isinstance(self.loss_weights["VQ"], Iterable)
                    else self.loss_weights["VQ"],
                    name=f"VQ{i + 1}",
                )
            ]

            self.post_vq += [Reshape(self.post_vq_shape[i], name=f"post_vq{i + 1}")]

            self.histcounters += [
                Histcounts(
                    value_range=[0, self.num_embeddings[i]],
                    nbins=self.num_embeddings[i],
                    name=f"hist{i + 1}",
                )
            ]

            self.fc_blocks += [
                fc_block(
                    self.pre_vq_shape[i],
                    self.num_fc_layers,
                    self.num_fc_units,
                    self.num_fc_output_classes,
                    dropout_rate=self.dropout_rate,
                    name=f"fc{i + 1}",
                )
            ]

            if i > 0:
                self.mse_lyrs += [SimpleMSE(name=f"mselyr{i}")]
                self.upconcat_list += [
                    UpConcat2D(
                        self.decoders[-i - 1].input_shape[1:-1],
                        mergetype="cat",
                        interpolation="bilinear",
                        name=f"upcat2d{self.num_encs - i}",
                    )
                ]

        b4_vqs = []
        for i in range(self.num_encs):
            enc = self.encoders[i](self.img_input if i == 0 else enc)
            b4_vqs.append(enc)

        self.quantized_tensors = []
        self.quantized_indices = []
        self.quantized_hist = []
        self.fc_outputs = []
        self.post_vq_tensors = []
        self.decoded_list = []
        self.qtvecs = []
        self.qtinds = []
        for i in range(self.num_encs):
            # connection is built from inside to outside
            b4_vq = b4_vqs[-i - 1]
            if i > 0:
                b4_vq = self.mse_lyrs[-i + 1]([b4_vq, dec])
            pre_vq_tsr = self.pre_vq[-i - 1](b4_vq)
            qtvec, qtind = self.vq_lyrs[-i - 1](pre_vq_tsr)
            self.qtvecs.append(qtvec)
            self.qtinds.append(qtind)
            qtvec_rshp = self.post_vq[-i - 1](self.qtvecs[-1])
            if self.num_encs - i in self.kwargs.get("zero_vqvec_idx"):
                print(f"vqvec{self.num_encs - i} is set to all zeros")
                qtvec_rshp = tfv1.zeros_like(qtvec_rshp)
            self.post_vq_tensors.append(qtvec_rshp)
            # count histograms of VQ code in each image
            emb_ind = Flatten(name=f"flathist{i + 1}")(qtind)
            self.quantized_hist.append(self.histcounters[-i - 1](emb_ind))
            self.quantized_tensors.append(qtvec)
            self.quantized_indices.append(qtind)
            if i > 0:
                qtvecup = self.upconcat_list[-i](self.post_vq_tensors)
            else:
                qtvecup = qtvec_rshp
            dec = self.decoders[-i - 1](qtvecup)
            self.decoded_list.append(dec)
            # connect an fc block after the VQ layer
            self.fc_outputs.append(self.fc_blocks[-i - 1](self.quantized_tensors[i]))
        # insert an fc_block to the top if fc_layer_connections 0 is selected,
        # which connects fc to the concatenated tensor right before dec
        if 0 in self.fc_layer_connections:
            self.fc_blocks = [
                fc_block(
                    [K.int_shape(i)[1:] for i in self.quantized_tensors],
                    self.num_fc_layers,
                    self.num_fc_units,
                    self.num_fc_output_classes,
                    name=f"fc0",
                )
            ] + self.fc_blocks
            self.fc_outputs = [
                self.fc_blocks[0](self.quantized_tensors)
            ] + self.fc_outputs
        self.model = Model(
            self.img_input,
            [dec] + [self.fc_outputs[-i] for i in self.fc_layer_connections],
        )

    def compile_model(self, data_variance):
        """
        Compile a TensorFLow model.
        :param data_variance: variance of training data. (required for VQ computation)
        """
        if self.model is None:
            raise ValueError("Model has not been created yet.")
        else:
            metrics_list = ["mse"]
            for i in range(self.num_encs):
                vql = self.vq_lyrs[i]
                metrics_list.append(Metric(vql.perplexity, name=f"prplx{i + 1}"))
                metrics_list.append(Metric(vql.e_latent_loss, name=f"e_loss{i + 1}"))
                metrics_list.append(Metric(vql.q_latent_loss, name=f"q_loss{i + 1}"))
            for i, m in enumerate(self.mse_lyrs):
                metrics_list.append(Metric(m.mse_loss, name=f"mse_mid{i + 1}"))
            metrics_list = [metrics_list] + (
                [["categorical_crossentropy", "accuracy"]]
                * len(self.fc_layer_connections)
            )

            self.model.compile(
                loss=[normalized_mse(var=data_variance)]
                + ["categorical_crossentropy"] * len(self.fc_layer_connections),
                optimizer=tfv1.keras.optimizers.Adam(lr=self.learn_rate),
                metrics=metrics_list,
                loss_weights=[self.loss_weights["decoder"]]
                + [self.loss_weights["fc"]] * len(self.fc_layer_connections),
            )

    def train_with_datamanager(
        self, datamanager, batch_size, max_epoch=100, col=0, **kwargs
    ):
        """
        Train model with data manager.
        :param datamanager: data manager object
        :param batch_size: batch size
        :param max_epoch: max epochs
        :param col: col argument of get_label_onehot
        """
        if (
            datamanager.train_generator is None
            or datamanager.val_generator is None
            or datamanager.test_generator is None
        ):
            datamanager.make_generators(
                batch_size, n_label_out=self.num_fc_layers, col=col
            )
        self.train_model(
            datamanager.train_generator,
            datamanager.val_generator,
            datamanager.test_generator,
            train_data_len=datamanager.train_data.shape[0],
            val_data_len=datamanager.val_data.shape[0],
            test_data_len=datamanager.test_data.shape[0],
            batch_size=batch_size,
            max_epoch=max_epoch,
            **kwargs,
        )

    def plot_history(self, history_df=None, title="Training history", savepath=None):
        """
        Plot training history.
        :param history_df: training history in dataframe format
        :param title: plot title
        :param savepath: save path
        """
        if history_df is None:
            if self.history_df is not None:
                history_df = self.history_df
            else:
                raise ValueError("No history_df found.")
        n_row, n_col = 2, 4
        fig, ax = plt.subplots(n_row, n_col, figsize=(6.5 * n_col, 5.5 * n_row))
        for i in range(n_row * n_col):
            ix, iy = i // n_col, i % n_col
            if i == 0:
                ax[ix, iy].text(
                    -0.1, 0.2, title, fontsize=18,
                )
                ax[ix, iy].axis("off")
            else:
                if i == 1:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        "loss",
                        title="Overall loss",
                        xlabel=None,
                        ylabel2=None,
                    )
                elif i == 2:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"decoder1_prplx{i + 1}" for i in range(self.num_encs)],
                        title="Perplexity",
                        xlabel=None,
                        ylabel1=None,
                        ylabel2=None,
                    )
                elif i == 3:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [
                            f"fc{i}_categorical_crossentropy"
                            for i in self.fc_layer_connections
                        ],
                        title="Classification loss",
                        xlabel=None,
                        ylabel1=None,
                    )
                elif i == 4:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"fc{i}_acc" for i in self.fc_layer_connections],
                        title="Classification accuracy",
                        ylabel1="accuracy",
                        ylabel2=None,
                    )
                elif i == 5:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"decoder1_e_loss{i + 1}" for i in range(self.num_encs)],
                        title="Commitment loss",
                        ylabel2=None,
                    )
                elif i == 6:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"decoder1_q_loss{i + 1}" for i in range(self.num_encs)],
                        title="Quantization loss",
                        ylabel1=None,
                        ylabel2=None,
                    )
                else:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        ["decoder1_mean_squared_error"]
                        + [
                            f"decoder1_mse_mid{i + 1}"
                            for i in range(len(self.mse_lyrs))
                        ],
                        title="Reconstruction loss",
                        ylabel1=None,
                    )
        plt.tight_layout()
        if savepath is None:
            savepath = self.savepath_dict["history"]
        plt.savefig(os.path.join(savepath, "train_history.png"), dpi=300)

    def plot_reconstruction_panel(
        self, train_raw, test_raw, channel_names=None, savepath=None, grid_size=(4, 8),
    ):
        """
        Plot input and output images in a panel.
        :param train_raw: training data
        :param test_raw: test data
        :param channel_names: a list of channel names
        :param savepath: save path
        :param grid_size: grid size for each panel
        """
        num_demo = min(train_raw.shape[0], test_raw.shape[0])
        train_raw = train_raw[:num_demo]
        test_raw = test_raw[:num_demo]
        train_gen = self.model.predict(train_raw, verbose=1)
        test_gen = self.model.predict(test_raw, verbose=1)
        if savepath is None:
            savepath = self.savepath_dict["history"]
        if channel_names is None or len(channel_names) != train_raw.shape[-1]:
            channel_names = [f"Ch{i}" for i in range(train_raw.shape[-1])]
        for i in range(train_raw.shape[-1]):
            plot_reconstruction_panel(
                train_raw[..., i],
                test_raw[..., i],
                train_gen[0][..., i],
                test_gen[0][..., i],
                savepath,
                channel_names[i],
                grid_size,
            )

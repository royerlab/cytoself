import os
import math
import traceback
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tfv1
from tensorflow.compat.v1.keras.layers import Flatten, Reshape
from tensorflow.compat.v1.keras.models import Model
from tqdm import tqdm

from cytoself.models.base import CytoselfBaseModel
from cytoself.external.efficientnet2d import EfficientEncoderB0
from cytoself.external.layers.vq import VectorQuantizer
from cytoself.components.layers.norm_mse import normalized_mse
from cytoself.components.layers.loss import SimpleMSE
from cytoself.components.metrics.metric import Metric
from cytoself.components.layers.upconcat import UpConcat2D
from cytoself.components.decoders.resnet import Decoderres_bicub
from cytoself.analysis.plot_functions.plot_histories import plothistory
from cytoself.utils.path_utils import split_path_all
from cytoself.components.layers.histcounts import Histcounts


class CytoselfSQModel(CytoselfBaseModel):
    """
    A model object for EfficientNet-ResNet VQVAE with split quantization but without pretext identification.
    """

    def __init__(
        self,
        input_image_shape,
        output_image_shape,
        q_splits,
        output_dir,
        suffix_dirs="",
        encoder_block_args=None,
        learn_rate=1e-4,
        early_stop_lr=1e-8,
        callbacks_start_from=10,
        num_embeddings=(128, 128),
        embedding_dims=(64, 64),
        commitment_costs=(0.25, 0.25),
        num_residual_layers=2,
        activation="swish",
        early_stop_patience=12,
        reduce_lr_patience=4,
        loss_weights={"VQ": 1},
        use_depthwise=False,
        **kwargs,
    ):
        """
        :param input_image_shape: dimensions of input image
        :param output_image_shape:  dimensions of output image
        :param q_splits: a list of number of splits in each VQ layer
        :param output_dir: base path for saving
        :param suffix_dirs: suffix for saving directory (custom comments on the directory)
        :param encoder_block_args: dict of params for encoder (EfficientNet)
        :param learn_rate: learn rate
        :param early_stop_lr: early stop learn rate
        :param callbacks_start_from: epoch number where callbacks start monitoring
        :param num_embeddings: number of codes in the codebooks in each VQ layer
        :param embedding_dims: dimensions of embeddings in each VQ layer
        :param commitment_costs: commitment cost in each VQ layer
        :param num_residual_layers: number of residual layers in decoders
        :param activation: activation function in all activation layers
        :param early_stop_patience: early stop patience
        :param reduce_lr_patience: Reduce learn rate patience
        :param loss_weights: a dict of coeff. to balance loss weights of multiple model outputs.
        :param use_depthwise: use depthwise conv in decoder if True
        """
        kwargs.setdefault("init_savepath", True)
        kwargs.setdefault("makedirs", True)
        kwargs.setdefault("do_print_model", False)
        kwargs.setdefault("path_dict", None)
        super().__init__(
            input_image_shape,
            output_image_shape,
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
            **kwargs,
        )
        self.use_depthwise = use_depthwise
        self.q_splits = q_splits
        self.pre_vq_shape = []
        self.post_vq_shape = []
        self.vqvec_slices_shape = []
        self.post_vq = []

        # Organize and create saving directories
        self.init_savepath(
            path_dict=self.kwargs.get("path_dict"),
            domakedirs=self.kwargs.get("makedirs"),
        )
        self.get_extended_zdim()

        # Construct a model
        self.construct_model()
        if self.kwargs.get("do_print_model"):
            try:
                self.plot_model()
                print("Model architecture is saved.")
            except Exception as e:
                print(traceback.format_exc())
                print("Model architecture was not saved.")
        if "mainpath" in self.savepath_dict:
            print(
                f'\n\nAll files will be saved at {self.savepath_dict["mainpath"]}\n\n'
            )
        else:
            print(
                "mainpath is not defined. Some default saving paths are also not likely to be defined.\n"
                "Please manually define savepath or some functions cannot save outputs with default settings."
            )

    def init_savepath(self, path_dict=None, domakedirs=True):
        if self.kwargs.get("init_savepath"):
            # One can create custom structured folder hierarchy here.
            self.savepath_dict["metadata"] = ""
            self.mkdirs_savepath_dict(path_dict=path_dict, makedirs=domakedirs)

    def get_extended_zdim(self):
        self.pre_vq_shape = []
        self.post_vq_shape = []
        self.vqvec_slices_shape = []
        for i, n_sp in enumerate(self.q_splits):
            ncol = math.ceil(math.sqrt(n_sp))
            nrow = math.ceil(n_sp / ncol)
            if ncol * nrow != n_sp:
                ncol, nrow = n_sp, 1
            z_dim_ex = [
                self.z_dims[i][0] * nrow,
                self.z_dims[i][1] * ncol,
                self.z_dims[i][2],
            ]
            z_dim_exrshp = self.z_dims[i][:-1] + [self.z_dims[i][-1] * n_sp]
            self.pre_vq_shape.append(z_dim_ex)
            self.post_vq_shape.append(z_dim_exrshp)
            self.vqvec_slices_shape.append(
                self.z_dims[i][:-1] + [n_sp, self.z_dims[i][-1]]
            )

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

            if i > 0:
                self.mse_lyrs += [SimpleMSE(name=f"mselyr{i}")]

        b4_vqs = []
        for i in range(self.num_encs):
            enc = self.encoders[i](self.img_input if i == 0 else enc)
            b4_vqs.append(enc)

        self.quantized_tensors = []
        self.quantized_indices = []
        self.quantized_hist = []
        post_vq_tsr = []
        for i in range(self.num_encs):
            # connection is built from inside to outside
            b4_vq = b4_vqs[-i - 1]
            if i > 0:
                b4_vq = self.mse_lyrs[-i + 1]([b4_vq, dec])
            pre_vq_tsr = self.pre_vq[-i - 1](b4_vq)
            qtvec, qtind = self.vq_lyrs[-i - 1](pre_vq_tsr)
            qtvec_rshp = self.post_vq[-i - 1](qtvec)
            post_vq_tsr.append(qtvec_rshp)
            # count histograms of VQ code in each image
            emb_ind = Flatten(name=f"flathist{i + 1}")(qtind)
            self.quantized_hist.append(self.histcounters[-i - 1](emb_ind))
            self.quantized_tensors.append(qtvec)
            self.quantized_indices.append(qtind)
            if i > 0:
                target_dim = self.decoders[-i - 1].input_shape
                qtvecup = UpConcat2D(
                    target_dim[1:-1], post_vq_tsr, name=f"upcat2d{self.num_encs - i}"
                )
            else:
                qtvecup = qtvec_rshp
            dec = self.decoders[-i - 1](qtvecup)
        self.model = Model(self.img_input, dec)

    def compile_model(self, data_variance):
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

            self.model.compile(
                loss=normalized_mse(var=data_variance),
                optimizer=tfv1.keras.optimizers.Adam(lr=self.learn_rate),
                metrics=metrics_list,
            )

    def train_model(
        self,
        train_data_generator,
        val_data_generator,
        test_data_generator,
        train_data_len,
        val_data_len,
        test_data_len,
        batch_size=32,
        max_epoch=900,
        reset_callbacks=True,
        **kwargs,
    ):
        """
        :param train_data_generator: data generator for training
        :param val_data_generator: data generator for validation
        :param test_data_generator: test data generator
        :param train_data_len: training data length
        :param val_data_len: validation data length
        :param test_data_len: test data length
        :param batch_size: batch size
        :param max_epoch: maximum epochs
        :param reset_callbacks: callbacks will be reset if True
        :return:
        """
        kwargs.setdefault("initial_epoch", 0)
        self.batch_size = batch_size
        if self.model is None:
            raise ValueError("Model has not been created yet.")
        else:
            if self.model.optimizer is None:
                raise ValueError("Model has not been compiled yet.")
            else:
                if reset_callbacks or self.callbacks is None:
                    self.init_callbacks()
                self.history = self.model.fit(
                    train_data_generator,
                    steps_per_epoch=(math.ceil(train_data_len / self.batch_size)),
                    epochs=max_epoch,
                    callbacks=self.callbacks,
                    validation_data=val_data_generator,
                    validation_steps=(math.ceil(val_data_len / self.batch_size)),
                    initial_epoch=kwargs["initial_epoch"],
                )
                self.post_fit(test_data_generator, test_data_len)

    def plot_history(self, history_df=None, title="Training history", savepath=None):
        if history_df is None:
            if self.history_df is not None:
                history_df = self.history_df
            else:
                raise ValueError("No history_df found.")
        n_row, n_col = 2, 3
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
                        [f"prplx{i + 1}" for i in range(self.num_encs)],
                        title="Perplexity",
                        xlabel=None,
                        ylabel1=None,
                    )
                elif i == 3:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"e_loss{i + 1}" for i in range(self.num_encs)],
                        title="Commitment loss",
                        ylabel2=None,
                    )
                elif i == 4:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        [f"q_loss{i + 1}" for i in range(self.num_encs)],
                        title="Quantization loss",
                        ylabel1=None,
                        ylabel2=None,
                    )
                else:
                    plothistory(
                        history_df,
                        ax[ix, iy],
                        ["mean_squared_error"]
                        + [f"mse_mid{i + 1}" for i in range(len(self.mse_lyrs))],
                        title="Reconstruction loss",
                        ylabel1=None,
                    )
        plt.tight_layout()
        if savepath is None:
            savepath = self.savepath_dict["history"]
        plt.savefig(os.path.join(savepath, "train_history.png"), dpi=300)

    def plot_vqind_channel(
        self, images, savepath=None, filename="vqind_perch", dpi=300
    ):
        if self.model_vq_ind is None:
            self.construct_vq_model("ind")
        if savepath == "default":
            savepath = self.savepath_dict["ft"]
        vqinds = self.model_vq_ind.predict(
            images, batch_size=self.batch_size, verbose=1
        )
        if not isinstance(vqinds, list):
            vqinds = [vqinds]

        for i, v in enumerate(vqinds):
            v = v.reshape((-1, self.z_dims[i][0], self.z_dims[i][1], self.q_splits[i]))
            nrow = self.q_splits[i] + images.shape[-1]
            ncol = images.shape[0]
            n_imch = images.shape[-1]
            f = plt.figure(figsize=(3 * ncol, 2.5 * nrow), constrained_layout=True)
            gs = f.add_gridspec(nrow, ncol)
            for j, im in enumerate(tqdm(images)):
                for ci in range(n_imch):
                    ax = f.add_subplot(gs[ci, j])
                    ax.imshow(
                        im[..., ci], cmap="gray",
                    )
                    ax.axis("off")
                for ci in range(self.q_splits[i]):
                    ax = f.add_subplot(gs[ci + n_imch, j])
                    ax.imshow(v[j, ..., ci], vmin=0, vmax=self.num_embeddings[i])
                    ax.axis("off")
            if savepath:
                f.savefig(os.path.join(savepath, f"{filename}_{i + 1}.png"), dpi=dpi)

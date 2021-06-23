import os
import math
import traceback
from collections.abc import Iterable
from warnings import warn

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tfv1
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.losses import MSE

from cytoself.components.utils.history_utils import merge_history, extract_history
from cytoself.external.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from cytoself.analysis.plot_functions.plot_reconstruction_panel import (
    plot_reconstruction_panel,
)


class CytoselfBaseModel:
    """
    Base class for a model object.
    """

    def __init__(
        self,
        input_image_shape,
        output_image_shape,
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
        early_stop_patience=8,
        reduce_lr_patience=4,
        loss_weights={"VQ": 1},
        **kwargs,
    ):
        """
        :param input_image_shape: dimensions of input image
        :param output_image_shape:  dimensions of output image
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
        """
        self.input_image_shape = input_image_shape
        self.output_image_shape = output_image_shape
        self.savepath_dict = {"basepath": output_dir}
        self.encoder_block_args = encoder_block_args
        self.learn_rate = learn_rate
        self.early_stop_lr = early_stop_lr
        self.callbacks_start_from = callbacks_start_from
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.commitment_costs = commitment_costs
        self.num_residual_layers = num_residual_layers
        self.activation = activation
        self.early_stop_patience = early_stop_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.basepath = "/share/LeonettiLab/opencell"
        self.suffix_dirs = suffix_dirs
        self.kwargs = kwargs

        self.batch_size = 32
        self.early_stop_mindelta = 0
        self.reduce_lr_factor = 0.1
        self.img_input = Input(self.input_image_shape, name="img_input")
        self.model = None
        self.model_vq = None
        self.model_vq_ind = None
        self.model_vq_hist = None
        self.quantized_tensors = []
        self.quantized_indices = []
        self.quantized_hist = []
        self.encoders = []
        self.decoders = []
        self.histcounters = []
        self.pre_vq = []
        self.vq_lyrs = []
        self.mse_lyrs = []
        self.upconcat_list = []

        self.early_stop_callback = None
        self.reduce_lr_callback = None
        self.checkpoint_callback = None
        self.callbacks = None

        self.history = None  # history obj from keras
        self.history_df = None  # DataFrame ready to plot and save

        self.embvec = None
        self.embind = None
        self.embindhist = None

        self.post_vq = []
        self.post_vq_tensors = []
        self.decoded_list = []
        self.qtvecs = []
        self.qtinds = []

        # Check loss_weights items
        if isinstance(loss_weights, dict):
            self.loss_weights = loss_weights
        else:
            self.loss_weights = {}
        for key in ["decoder", "fc", "VQ"]:
            if key not in self.loss_weights:
                self.loss_weights.update({key: 1})

        # Determine encoder block arguments
        if self.encoder_block_args is None:
            encoder_block_args1 = [
                {
                    "kernel_size": 3,
                    "repeats": 1,
                    "filters_in": 32,
                    "filters_out": 16,
                    "expand_ratio": 1,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 3,
                    "repeats": 2,
                    "filters_in": 16,
                    "filters_out": 24,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 2,
                    "filters_in": 24,
                    "filters_out": 40,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
            ]
            encoder_block_args2 = [
                {
                    "kernel_size": 3,
                    "repeats": 3,
                    "filters_in": 40,
                    "filters_out": 80,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 3,
                    "filters_in": 80,
                    "filters_out": 112,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 5,
                    "repeats": 4,
                    "filters_in": 112,
                    "filters_out": 192,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 2,
                    "se_ratio": 0.25,
                },
                {
                    "kernel_size": 3,
                    "repeats": 1,
                    "filters_in": 192,
                    "filters_out": 320,
                    "expand_ratio": 6,
                    "id_skip": True,
                    "strides": 1,
                    "se_ratio": 0.25,
                },
            ]
            self.encoder_block_args = [encoder_block_args1, encoder_block_args2]
        self.num_encs = len(self.encoder_block_args)

        # Automatically detect number of scale down in each encoder
        scale_down_all = [1]
        for args in self.encoder_block_args:
            scale_down1 = scale_down_all[-1]
            for i in args:
                if i["strides"] == 2:
                    scale_down1 += 1
            scale_down_all.append(scale_down1)
        scale_down_all = scale_down_all[1:]

        # Calculate the intermediate output sizes.
        self.z_dims = []
        for i, sd in enumerate(scale_down_all):
            self.z_dims.append(
                [math.ceil(i / 2 ** sd) for i in self.input_image_shape[:-1]]
                + [self.embedding_dims[i]]
            )

    def init_savepath(self):
        NotImplemented

    def mkdirs_savepath_dict(self, path_dict=None, makedirs=True):
        """
        Create saving directories.
        """
        if "metadata" in self.savepath_dict:
            self.savepath_dict["mainpath"] = os.path.join(
                self.savepath_dict["basepath"], self.savepath_dict["metadata"]
            )
            self.savepath_dict["checkpoints"] = os.path.join(
                self.savepath_dict["mainpath"], "checkpoints"
            )
            self.savepath_dict["umaps"] = os.path.join(
                self.savepath_dict["mainpath"], "umaps"
            )
            self.savepath_dict["history"] = os.path.join(
                self.savepath_dict["mainpath"], "history"
            )
            self.savepath_dict["emb"] = os.path.join(
                self.savepath_dict["mainpath"], "embeddings"
            )
            self.savepath_dict["ft"] = os.path.join(
                self.savepath_dict["mainpath"], "ft_analysis"
            )
        if path_dict is not None:
            self.savepath_dict.update(path_dict)
        if makedirs:
            for key, val in self.savepath_dict.items():
                if key != "metadata":
                    os.makedirs(val, exist_ok=True)

    def construct_model(self):
        """
        Construct a VQVAE model
        """
        NotImplemented

    def compile_model(self, data_variance):
        NotImplemented

    def compile_with_datamanager(self, datamanager):
        """
        Compile model with data manager.
        :param datamanager: data manager object
        """
        if len(datamanager.train_data) > 0:
            self.compile_model(np.var(datamanager.train_data))
        else:
            print("No train_data found in datamanager.")

    def load_model(self, path=None):
        if path is None:
            candidates = []
            nepochs = []
            for i in os.scandir(self.savepath_dict["mainpath"]):
                if i.name[-2:] == "h5":
                    p = i.path
                    nepochs.append(int(p[p.find("_ep") + 3 : p.find(".h5")]))
                    candidates.append(p)
            path = candidates[np.argmax(nepochs)]
        self.model.load_weights(path)
        print(f"{path}\nis loaded.")

    def init_callbacks(self):
        self.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=self.early_stop_mindelta,
            patience=self.early_stop_patience,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
            exclude_initial_epochs=self.callbacks_start_from,
        )
        self.reduce_lr_callback = ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.reduce_lr_factor,
            verbose=1,
            patience=self.reduce_lr_patience,
            mode="auto",
            min_lr=self.early_stop_lr,
            exclude_initial_epochs=self.callbacks_start_from,
        )
        self.model_checkpoint_callback = ModelCheckpoint(
            os.path.join(
                self.savepath_dict["checkpoints"], "model_weights.{epoch:04d}.h5"
            ),
            monitor="val_loss",
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            exclude_initial_epochs=self.callbacks_start_from,
        )
        self.callbacks = [
            self.early_stop_callback,
            self.reduce_lr_callback,
            self.model_checkpoint_callback,
        ]

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
        NotImplemented

    def plot_history(self, history_df=None, title="", savepath=None):
        NotImplemented

    def combine_history(self, new_history_df):
        if self.history_df is None:
            self.history_df = new_history_df
        else:
            self.history_df = merge_history(self.history_df, new_history_df)

    def plot_reconstruction_panel(
        self, train_raw, test_raw, channel_names=None, savepath=None, grid_size=(4, 8),
    ):
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
                train_gen[..., i],
                test_gen[..., i],
                savepath,
                channel_names[i],
                grid_size,
            )

    def plot_model(self, path=None, expand_nested=False, dpi=200):
        if path is None:
            path = self.savepath_dict["history"]
        tfv1.keras.utils.plot_model(
            self.model,
            os.path.join(path, "model.png"),
            True,
            expand_nested=expand_nested,
            dpi=dpi,
        )

    def save_vq_codebook(self):
        if not isinstance(self.vq_lyrs, Iterable):
            self.vq_lyrs = [self.vq_lyrs]
        for i, vq in enumerate(self.vq_lyrs):
            cb = K.eval(vq.embeddings)
            np.save(
                os.path.join(self.savepath_dict["emb"], f"codebook_vq{i + 1}.npy"), cb
            )

    def post_fit(self, test_data_generator, test_data_len):
        """
        A common of processes after model.fit
        :param test_data_generator: data generator for test data
        :param test_data_len: test data length
        """
        score = self.model.evaluate(
            test_data_generator,
            steps=math.ceil(test_data_len / self.batch_size),
            verbose=1,
        )
        self.post_evaluate(score)

    def post_evaluate(self, score):
        """
        A common of processes after model.evaluate
        :param test_data_generator: data generator for test data
        :param test_data_len: test data length
        """
        self.history_df, numepo = extract_history(self.history, score, self.model)
        pd.DataFrame(self.history_df).to_csv(
            os.path.join(self.savepath_dict["history"], "train_history.csv"),
            index=False,
        )
        self.plot_history()
        try:
            self.model.save_weights(
                os.path.join(self.savepath_dict["mainpath"], f"model_ep{numepo:04d}.h5")
            )
        except Exception as e:
            print(traceback.format_exc())
            print("Model was not saved.")
        # self.save_vq_codebook()
        # print("VQ codebook is saved.")

    def construct_vq_model(self, modeltype):
        if self.model is None:
            raise ValueError("Model has not been created yet.")
        else:
            if modeltype == "vec":
                self.model_vq = Model(self.img_input, self.quantized_tensors[::-1])
            elif modeltype == "ind":
                self.model_vq_ind = Model(self.img_input, self.quantized_indices[::-1])
            elif modeltype == "indhist":
                self.model_vq_hist = Model(self.img_input, self.quantized_hist[::-1])
            else:
                print("modeltype only accepts vec, ind or indhist.")

    def construct_decoder_model(self, dec_idx=1, eval_mode=None):
        """
        Construct a decoder model
        :param dec_idx: decoder index; starting from 1 at the shallowest vq
        :param eval_mode: evaluation mode; None: only outputs decoded img, mse: dec img + mse with original img
        :return: decoder model with input shape designed for unexpanded latent vector
        """
        if self.model is None:
            raise ValueError("Model has not been created yet.")
        else:
            n_inputs = self.num_encs - dec_idx + 1
            if n_inputs == 1:
                input_list = Input(
                    self.post_vq_tensors[self.num_encs - dec_idx].shape[1:],
                    name=f"vqvec{dec_idx}",
                )
                qtvecup = input_list
            else:
                input_list = [
                    Input(self.post_vq_tensors[i].shape[1:], name=f"vqvec{i + 1}")
                    for i in range(n_inputs)
                ]
                qtvecup = self.upconcat_list[dec_idx - 1](input_list)
            dec = self.decoders[dec_idx - 1](qtvecup)

            if eval_mode == "mse":
                mse_out = tfv1.math.reduce_sum(
                    MSE(self.img_input, dec), axis=[1, 2], name="mse_sum"
                )
                if isinstance(input_list, list):
                    input_list = [input_list]
                return Model(input_list + [self.img_input], [dec, mse_out])
            else:
                return Model(input_list, dec)

    def construct_decoder_model_one_input(self, dec_idx=1):
        """
        Construct a decoder model that has only one input
        :param dec_idx: decoder index; starting from 1 at the shallowest vq
        :return: decoder model with input shape designed for unexpanded latent vector
        """
        if self.model is None:
            raise ValueError("Model has not been created yet.")
        else:
            post_vq_tensors = []
            for i in range(self.num_encs):
                if i + 1 == dec_idx:
                    input_lyr = Input(
                        self.qtvecs[-i - 1].shape[1:], name=f"vqvec{dec_idx}"
                    )
                    qtvec_rshp = self.post_vq[i](input_lyr)
                else:
                    qtvec_rshp = tfv1.zeros(
                        tfv1.stack(
                            [
                                tfv1.shape(input_lyr)[0],
                                *K.int_shape(self.post_vq_tensors[-i - 1])[1:],
                            ],
                            name="shape_stack",
                        ),
                        name="zeros",
                    )
                post_vq_tensors.append(qtvec_rshp)
            if dec_idx - 1 < self.num_encs:
                qtvecup = self.upconcat_list[dec_idx - 1](post_vq_tensors[::-1])
            else:
                qtvecup = post_vq_tensors[-1]
            dec = self.decoders[-i - 1](qtvecup)
            return Model(input_lyr, dec)

    def calc_embvec(self, data, savepath=None, filename="test_vqvec", do_return=False):
        """
        Calculate the VQ embedding vectors
        """
        if self.model_vq is None:
            self.construct_vq_model("vec")
        print("Inferring embedding vectors...")
        embvec = self.model_vq.predict(data, batch_size=self.batch_size * 4, verbose=1)
        if not isinstance(embvec, list):
            embvec = [embvec]
        if savepath:
            if savepath == "default":
                savepath = self.savepath_dict["emb"]
            [
                np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                for i, v in enumerate(embvec)
            ]
        if do_return:
            return embvec
        else:
            self.embvec = embvec

    def calc_embind(self, data, savepath=None, filename="test_vqind", do_return=False):
        """
        Calculate the VQ embedding indicies
        """
        if self.model_vq_ind is None:
            self.construct_vq_model("ind")
        print("Inferring embedding indices...")
        embind = self.model_vq_ind.predict(
            data, batch_size=self.batch_size * 4, verbose=1
        )
        if not isinstance(embind, list):
            embind = [embind]
        if savepath:
            if savepath == "default":
                savepath = self.savepath_dict["emb"]
            [
                np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                for i, v in enumerate(embind)
            ]
        if do_return:
            return embind
        else:
            self.embind = embind

    def calc_embindhist(
        self, data, savepath=None, filename="test_vqindhist", do_return=False
    ):
        """
        Calculate the histogram of VQ embedding indicies
        """
        if self.model_vq_hist is None:
            self.construct_vq_model("indhist")
        print("Inferring embedding histogram...")
        embindhist = self.model_vq_hist.predict(
            data, batch_size=self.batch_size * 4, verbose=1
        )
        if not isinstance(embindhist, list):
            embindhist = [embindhist]
        if savepath:
            if savepath == "default":
                savepath = self.savepath_dict["emb"]
            [
                np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                for i, v in enumerate(embindhist)
            ]
        if do_return:
            return embindhist
        else:
            self.embindhist = embindhist

    def calc_receptive_field(self):
        """
        Calculate receptive field of encoder in horizontal direction
        :return: mean receptive field size
        """
        if not self.encoders:
            warn("No encoder was found.")
        else:
            data = np.zeros((1,) + self.encoders[0].input_shape[1:])
            # rf_all = []
            # for enc in self.encoders:
            enc = self.encoders[0]
            resp0 = enc.predict(data)
            width = data.shape[2]
            resp = np.zeros((width, resp0.shape[2]))
            for j in range(width):
                data = np.zeros_like(data)
                data[0, j, j, :] = 1
                out = enc.predict(data)
                diff = out - resp0
                diffsum = diff.sum(axis=(0, 1, 3))
                resp[j] = diffsum
            rf = []
            for i in range(resp.shape[1]):
                if resp[0, i] == 0 and resp[-1, i] == 0:
                    ind = np.where(resp[:, i] != 0)[0]
                    rf.append(ind[-1] - ind[0] + 1)
            rf = np.mean(rf) if len(rf) > 0 else width
            # rf_all.append(rf)
            return rf

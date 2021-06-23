import math
import os
import pickle
from collections import Iterable, deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import umap
import colorcet as cc
import seaborn as sns
from warnings import warn

from cytoself.analysis.pearson_correlation import selfpearson_multi


class Analytics:
    """
    A class object for analyzing and visualizing data.
    """

    def __init__(self, model=None, data_manager=None, gt_table=None):
        """
        :param modelect: model object
        :param datamanager: data manager
        :param gt_table: ground true table
        """
        self.model = model
        self.data_manager = data_manager
        self.gt_table = gt_table
        self.vec_umap = None
        self.ind_umap = None
        self.indhist_umap = None
        self.model_vec_umap = []
        self.model_ind_umap = []
        self.model_indhist_umap = []
        self.heatmaps_all = []
        self.corr_idx_idx = []
        self.matrix_cellid_idx = []
        self.dendrogram_index = []
        if self.model is None or self.model.model is None:
            warn("No CNN model is built in modelect.", UserWarning)

    def calc_umap_embvec(
        self,
        data=None,
        selected_ind=None,
        target_vq_layer=None,
        savepath=None,
        filename="vqvec_umap",
        verbose=True,
    ):
        """
        Compute umap of embedding vectors.
        :param data: embedding vector data
        :param selected_ind: only selected feature index will be used to compute umap
        :param target_vq_layer: the vq layer to output embedding vector;
        1 for local representation, 2 for global representation
        :param savepath: save path
        :param filename: file name
        :param verbose: verbosity of umap.UMAP
        """
        if data is None:
            if self.model.embvec:
                data = self.model.embvec
            else:
                self.model.calc_embvec(self.data_manager.test_data)
                data = self.model.embvec
        else:
            if not isinstance(data, list):
                data = [data]
        if selected_ind is not None:
            data = [d[:, i] for d, i in zip(data, selected_ind)]
        if target_vq_layer:
            if isinstance(target_vq_layer, int):
                target_vq_layer = [target_vq_layer]
            data = [
                d if i + 1 in target_vq_layer else np.array([])
                for i, d in enumerate(data)
            ]
        if self.model_vec_umap != []:
            warn(
                "vqvec models is not empty. vqvec models will be overwritten.",
                UserWarning,
            )

        print(f"Computing UMAP...")
        self.model_vec_umap = []
        self.vec_umap = []
        for v in data:
            if min(v.shape) > 0:
                reducer = umap.UMAP(verbose=verbose)
                u = reducer.fit_transform(v.reshape(v.shape[0], -1))
            else:
                reducer = []
                u = np.array([])
            self.model_vec_umap.append(reducer)
            self.vec_umap.append(u)
        if savepath:
            if savepath == "default":
                savepath = self.model.savepath_dict["emb"]
            for i, v in enumerate(self.vec_umap):
                if min(v.shape) > 0:
                    np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                    try:
                        [
                            pickle.dump(
                                v,
                                open(
                                    os.path.join(
                                        savepath, f"model_{filename}{i + 1}.ump"
                                    ),
                                    "wb",
                                ),
                                protocol=4,
                            )
                            for i, v in enumerate(self.model_vec_umap)
                            if v != []
                        ]
                    except:
                        print("UMAP model was not saved.")

    def calc_umap_embind(
        self,
        data=None,
        selected_ind=None,
        target_vq_layer=None,
        savepath=None,
        filename="vqind_umap",
        verbose=True,
    ):
        """
        Compute umap of embedding index.
        :param data: embedding index data
        :param selected_ind: only selected feature index will be used to compute umap
        :param target_vq_layer: the vq layer to output embedding vector;
        1 for local representation, 2 for global representation
        :param savepath: save path
        :param filename: file name
        :param verbose: verbosity of umap.UMAP
        """
        if data is None:
            if self.model.embind:
                data = self.model.embind
            else:
                self.model.calc_embind(self.data_manager.test_data)
                data = self.model.embind
        else:
            if not isinstance(data, list):
                data = [data]
        if selected_ind is not None:
            data = [d[:, i] for d, i in zip(data, selected_ind)]
        if target_vq_layer:
            if isinstance(target_vq_layer, int):
                target_vq_layer = [target_vq_layer]
            data = [
                d if i + 1 in target_vq_layer else np.array([])
                for i, d in enumerate(data)
            ]
        if self.model_ind_umap != []:
            warn("vqind models is not empty. vqind models will be overwritten.")

        print(f"Computing UMAP...")
        self.model_ind_umap = []
        self.ind_umap = []
        for v in data:
            if min(v.shape) > 0:
                reducer = umap.UMAP(verbose=verbose)
                u = reducer.fit_transform(v.reshape(v.shape[0], -1))
            else:
                reducer = []
                u = np.array([])
            self.model_ind_umap.append(reducer)
            self.ind_umap.append(u)
        if savepath:
            if savepath == "default":
                savepath = self.model.savepath_dict["emb"]
            for i, v in enumerate(self.ind_umap):
                if min(v.shape) > 0:
                    np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                try:
                    [
                        pickle.dump(
                            v,
                            open(
                                os.path.join(savepath, f"model_{filename}{i + 1}.ump"),
                                "wb",
                            ),
                            protocol=4,
                        )
                        for i, v in enumerate(self.model_ind_umap)
                        if v != []
                    ]
                except:
                    print("UMAP model was not saved.")

    def calc_umap_embindhist(
        self,
        data=None,
        selected_ind=None,
        target_vq_layer=None,
        savepath=None,
        filename="vqindhist_umap",
        verbose=True,
    ):
        """
        Copute umap with vq index histogram
        :param data: embedding index histogram data
        :param selected_ind: only selected feature index will be used to compute umap
        :param target_vq_layer: the vq layer to output embedding vector;
        1 for local representation, 2 for global representation
        :param savepath: saving path
        :param filename: file name
        :param verbose: verbosity of umap.UMAP
        """
        if data is None:
            if self.model.embindhist:
                data = self.model.embindhist
            else:
                self.model.calc_embindhist(self.data_manager.test_data)
                data = self.model.embindhist
        else:
            if not isinstance(data, list):
                data = [data]
        if selected_ind is not None:
            data = [d[:, i] for d, i in zip(data, selected_ind)]
        if target_vq_layer:
            if isinstance(target_vq_layer, int):
                target_vq_layer = [target_vq_layer]
            data = [
                d if i + 1 in target_vq_layer else np.array([])
                for i, d in enumerate(data)
            ]
        if self.model_indhist_umap != []:
            warn(
                "vqindhist models is not empty. vqindhist models will be overwritten.",
                UserWarning,
            )

        print(f"Computing UMAP...")
        self.model_indhist_umap = []
        self.indhist_umap = []
        for v in data:
            if min(v.shape) > 0:
                reducer = umap.UMAP(verbose=verbose)
                u = reducer.fit_transform(v.reshape(v.shape[0], -1))
            else:
                reducer = []
                u = np.array([])
            self.model_indhist_umap.append(reducer)
            self.indhist_umap.append(u)
        if savepath:
            if savepath == "default":
                savepath = self.model.savepath_dict["emb"]
            for i, v in enumerate(self.indhist_umap):
                if min(v.shape) > 0:
                    np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                    try:
                        [
                            pickle.dump(
                                v,
                                open(
                                    os.path.join(
                                        savepath, f"model_{filename}{i + 1}.ump"
                                    ),
                                    "wb",
                                ),
                                protocol=4,
                            )
                            for i, v in enumerate(self.model_indhist_umap)
                            if v != []
                        ]
                    except:
                        print("UMAP model was not saved.")

    def transform_umap(
        self,
        data=None,
        model_type=None,
        selected_ind=None,
        savepath=None,
        filename="umap_transfered",
    ):
        """
        Convert input data to UMAP embedding; This is used when you already have a UMAP model.
        :param data: input data
        :param model_type: 'vec' or 'ind' or 'indhist'
        :param selected_ind: only run data with selected indices
        :param savepath: saving path
        :param filename: file name
        :return: umap embeddings
        """
        if not isinstance(data, list):
            data = [data]
        if model_type == "vec":
            model = self.model_vec_umap
        elif model_type == "ind":
            model = self.model_ind_umap
        elif model_type == "indhist":
            model = self.model_indhist_umap
        else:
            warn('Unknown model_type. Only "vec", "ind" or "indhist" is acceptable.')
        if model == []:
            print("model is empty. Load model or create a model first.")
        else:
            if len(data) != len(model):
                raise ValueError(
                    f"The number of datasets ({len(data)}) does not match with the number of models ({len(model)})."
                )
            if selected_ind is not None:
                data = [d[:, i] for d, i in zip(data, selected_ind)]
            results = []
            for v, m in zip(data, model):
                if min(v.shape) > 0:
                    u = m.transform(v.reshape(v.shape[0], -1))
                else:
                    u = np.array([])
                results.append(u)
            if savepath:
                if savepath == "default":
                    savepath = self.model.savepath_dict["emb"]
                [
                    np.save(os.path.join(savepath, f"{filename}{i + 1}.npy"), v)
                    for i, v in enumerate(results)
                ]
            return results

    def plot_umaps_gt(
        self,
        data=None,
        label=None,
        gt_table=None,
        savepath=None,
        target_vq_layer=2,
        filename="umap_gt",
        cmap="tab20",
        xlim=None,
        ylim=None,
        titles=None,
        subplot_shape=None,
    ):
        """
        Plot umaps by ground true labels
        :param data: vq index histogram data
        :param label: label
        :param gt_table: ground true table
        :param savepath: saving path
        :param target_vq_layer: 1 for local representation, 2 for global representation
        :param filename: file name
        :param cmap: color map
        :param xlim: x axis limits; [[low, high], [low, high]]
        :param ylim: y axis limits; [[low, high], [low, high]]
        :param titles: custom titles for each plot
        :param subplot_shape: custom subplot shape
        """
        if label is None:
            label = self.data_manager.test_label
        if gt_table is None:
            if self.gt_table is None:
                raise ValueError("gt_table is not provided.")
            else:
                gt_table = self.gt_table
        if savepath == "default":
            savepath = self.model.savepath_dict["umaps"]
        if isinstance(target_vq_layer, int):
            target_vq_layer = [target_vq_layer]

        gt_name = gt_table.iloc[:, 0]
        uniq_group = np.unique(gt_table.iloc[:, 1])

        # make sure data is in a list
        data = data if isinstance(data, list) else [data]
        n_subplots = 0
        for d in data:
            if min(d.shape) > 0:
                n_subplots += 1
        # get subplot shape
        if subplot_shape is None:
            ncol = math.ceil(math.sqrt(n_subplots))
            nrow = math.ceil(n_subplots / ncol)
        else:
            nrow, ncol = subplot_shape

        plt.figure(figsize=(6 * ncol, 5 * nrow))
        subplot_ind = 1
        for vqi, idht in enumerate(data):
            if vqi + 1 in target_vq_layer:
                print(f"Plotting {filename} subplot{vqi} ...")
                plt.subplot(nrow, ncol, subplot_ind)
                if min(idht.shape) > 0:
                    ind = np.isin(label[:, 0], gt_name)
                    # plot the layer of 'others'
                    sctrs = plt.scatter(
                        idht[~ind, 0],
                        idht[~ind, 1],
                        s=0.2,
                        c=np.array(cm.Greys(25)).reshape(1, -1),
                        alpha=0.1,
                        label="others",
                    )
                    # plot each cluster layer
                    for i, fname in enumerate(tqdm(uniq_group)):
                        group0 = gt_table[gt_table.iloc[:, 1] == fname]
                        ind = np.isin(label[:, 0], group0.iloc[:, 0])
                        data0 = idht[ind]
                        plt.scatter(
                            data0[:, 0],
                            data0[:, 1],
                            s=0.2,
                            c=np.array(cm.get_cmap(cmap).colors[i]).reshape(1, -1),
                            label=fname,
                        )
                    if xlim is not None and xlim[vqi] is not None:
                        plt.xlim(xlim[vqi])
                    if ylim is not None and ylim[vqi] is not None:
                        plt.ylim(ylim[vqi])

                hndls, names = sctrs.axes.get_legend_handles_labels()
                hndls = deque(hndls)
                names = deque(names)
                hndls.rotate(-1)
                names.rotate(-1)
                leg = plt.legend(
                    hndls,
                    names,
                    prop={"size": 6},
                    bbox_to_anchor=(1, 1),
                    loc="upper left",
                )
                for ll in range(len(names)):
                    leg.legendHandles[ll]._sizes = [6]
                if titles is None:
                    plt.title(f"Ground true samples vq{vqi + 1}")
                elif isinstance(titles, str):
                    plt.title(f"{titles} vq{vqi + 1}")
                else:
                    plt.title(titles[vqi])
                plt.ylabel("Umap 2")
                plt.xlabel("Umap 1")
                subplot_ind += 1
        plt.tight_layout()
        if savepath:
            plt.savefig(os.path.join(savepath, f"{filename}.png"), dpi=300)

    def calc_plot_umaps_gt(
        self,
        embedding_type,
        label=None,
        gt_table=None,
        savepath="default",
        target_vq_layer=2,
        filename="umap_gt",
        cmap="tab20",
        xlim=None,
        ylim=None,
        titles=None,
        subplot_shape=None,
    ):
        """
        Compute and plot umaps in one function.
        :param embedding_type: embedding type; 'vec' for vq vector, 'ind' for vq index, 'indhist' for vq index histogram
        :param label: label
        :param gt_table: ground true table
        :param savepath: saving path
        :param target_vq_layer: 1 for local representation, 2 for global representation
        :param filename: file name
        :param cmap: color map
        :param xlim: x axis limits; [[low, high], [low, high]]
        :param ylim: y axis limits; [[low, high], [low, high]]
        :param titles: custom titles for each plot
        :param subplot_shape: custom subplot shape
        """
        if embedding_type == "vec":
            if self.vec_umap is None:
                self.calc_umap_embvec(target_vq_layer=target_vq_layer)
            self.plot_umaps_gt(
                self.vec_umap,
                label=label,
                gt_table=gt_table,
                savepath=savepath,
                target_vq_layer=target_vq_layer,
                filename=filename,
                cmap=cmap,
                xlim=xlim,
                ylim=ylim,
                titles=titles,
                subplot_shape=subplot_shape,
            )
        elif embedding_type == "indhist":
            if self.indhist_umap is None:
                self.calc_umap_embindhist(target_vq_layer=target_vq_layer)
            self.plot_umaps_gt(
                self.ind_umap,
                label=label,
                gt_table=gt_table,
                savepath=savepath,
                target_vq_layer=target_vq_layer,
                filename=filename,
                cmap=cmap,
                xlim=xlim,
                ylim=ylim,
                titles=titles,
                subplot_shape=subplot_shape,
            )
        elif embedding_type == "ind":
            if self.ind_umap is None:
                self.calc_umap_embind(target_vq_layer=target_vq_layer)
            self.plot_umaps_gt(
                self.ind_umap,
                label=label,
                gt_table=gt_table,
                savepath=savepath,
                target_vq_layer=target_vq_layer,
                filename=filename,
                cmap=cmap,
                xlim=xlim,
                ylim=ylim,
                titles=titles,
                subplot_shape=subplot_shape,
            )

    def calc_cellid_vqidx(self, data=None, savepath=None):
        """
        Compute the matrixes of cell line id vs. vq index where the intensity values represent indhist per image.
        This is needed for computing feature heatmap.
        :param data: image data
        :param savepath: save path
        """
        if data is None:
            data = self.data_manager.test_data
        if self.model.embindhist is None:
            self.model.calc_embindhist(data)
        if not self.data_manager.n_classes:
            self.data_manager.get_unique_labels()

        print("Computing cellid vq index matrix...")
        self.matrix_cellid_idx = []
        for ii, idht in enumerate(self.model.embindhist):
            data_by_id = np.zeros((self.data_manager.n_classes, idht.shape[-1]))
            for i, id_ in enumerate(tqdm(self.data_manager.uniq_label)):
                data0 = idht[self.data_manager.test_label[:, 0] == id_]
                data_by_id[i, :] = data0.sum(0) / data0.shape[0]
            self.matrix_cellid_idx.append(data_by_id)
        self.matrix_cellid_idx = np.stack(self.matrix_cellid_idx)

        if savepath:
            np.save(
                os.path.join(savepath, "mtrx_cellid_vqidx.npy"), self.matrix_cellid_idx
            )

    def calc_corr_idx_idx(
        self, data, fileName="cellid_idx", num_cores=64, savepath=None
    ):
        """
        Calculate pearson's correlation between vq index vs. vq index.
        :param data: data
        :param fileName: file name
        :param num_cores: number of cores to use for parallel computation
        :param savepath: save path
        """
        if not isinstance(data, list):
            data = [data]

        print("Computing self Pearson correlation...")
        self.corr_idx_idx = []
        for d in data:
            if len(d) > 0:
                d = np.nan_to_num(selfpearson_multi(d, num_cores=num_cores))
            self.corr_idx_idx.append(d)
        # self.corr_idx_idx = np.stack(self.corr_idx_idx)
        if savepath:
            if savepath == "default":
                savepath0 = self.model.savepath_dict["ft"]
            else:
                savepath0 = savepath
            for i, d in enumerate(self.corr_idx_idx):
                np.save(os.path.join(savepath0, f"{fileName}_{i + 1}.npy"), d)

    def plot_clustermaps(
        self,
        data=None,
        corr_idx_idx=None,
        target_vq_layer=1,
        datatype="cellid_idx",
        savepath=None,
        filename="indhist_heatmap",
        format="png",
        num_cores=64,
    ):
        """
        Plot (hierarchical clustering) heatmaps of vqind vs. vqind
        :param data: matrix of cell id vs vq index or codebook
        :param corr_idx_idx: correlation of vq index vs. vq index
        :param target_vq_layer: 1 for local representation, 2 for global representation
        :param savepath: path to save heatmaps
        :param filename: file name
        :param format: save format; e.g. png, pdf
        :param datatype: cellid_idx or codebook
        :param num_cores: number of cores to use for parallel computation
        """
        if isinstance(target_vq_layer, int):
            target_vq_layer = [target_vq_layer]

        # check data
        if data is None:
            if datatype == "cellid_idx":
                if self.matrix_cellid_idx == []:
                    self.calc_cellid_vqidx()
                data = self.matrix_cellid_idx.copy()
            elif datatype == "codebook":
                if self.model.codebooks == []:
                    self.model.get_vq_codebook(save=False)
                data = self.model.codebooks.copy()

            # disable unselected vq layer
            data0 = []
            for i, d in enumerate(data):
                if i + 1 not in target_vq_layer:
                    d = np.array([])
                data0.append(d)
            data = data0
        if not isinstance(data, list):
            data = [data]

        # check corr_idx_idx
        if corr_idx_idx is None:
            if self.corr_idx_idx == []:
                self.calc_corr_idx_idx(
                    data=data,
                    fileName="corridx_" + datatype,
                    num_cores=num_cores,
                    savepath=savepath,
                )
            corr_idx_idx = self.corr_idx_idx

        # plot heatmaps
        self.heatmaps_all = []
        for ii, data in enumerate(corr_idx_idx):
            if len(data) > 0:
                print("computing clustermaps...")
                g = sns.clustermap(
                    np.nan_to_num(data),
                    cmap=cc.diverging_bwr_20_95_c54,
                    vmin=-1,
                    vmax=1,
                )
                g.ax_col_dendrogram.set_title(
                    f"vq{ii + 1} indhist Peason corr hierarchy link"
                )
                g.ax_heatmap.set_xlabel("vq index")
                g.ax_heatmap.set_ylabel("vq index")
                if savepath:
                    if savepath == "default":
                        savepath0 = self.model.savepath_dict["ft"]
                    else:
                        savepath0 = savepath
                    g.savefig(
                        os.path.join(savepath0, f"{filename}{ii + 1}.{format}"), dpi=300
                    )
            else:
                g = []
            self.heatmaps_all.append(g)

        self.dendrogram_index = [
            g.dendrogram_row.reordered_ind if g != [] else [] for g in self.heatmaps_all
        ]
        if savepath:
            if savepath == "default":
                savepath0 = self.model.savepath_dict["ft"]
            else:
                savepath0 = savepath
            for i, d in enumerate(self.dendrogram_index):
                if len(d) > 0:
                    np.save(
                        os.path.join(savepath0, f"{filename}_dgram_index{i + 1}.npy"), d
                    )
                    pickle.dump(
                        self.heatmaps_all[i],
                        open(os.path.join(savepath0, f"{filename}{i + 1}.hmp"), "wb"),
                        protocol=4,
                    )

    def load_dendrogram_index(self, filepath):
        """
        Load dendrogram index. Only npy format is accepted.
        :param filepath: file path, can be a list of file paths
        """
        if not isinstance(filepath, list):
            filepath = [filepath]
        self.dendrogram_index = [np.load(p) for p in filepath]

    def plot_feature_spectrum_from_image(
        self,
        data,
        target_vq_layer=1,
        take_mean=True,
        savepath=None,
        filename="Feature_spectrum",
        title=None,
    ):
        """
        Plot feature spectrum from image.
        :param data: image data; make sure it has 4 dimensions (i.e. batch, x, y, channel).
        :param target_vq_layer: 1 for local representation, 2 for global representation
        :param take_mean: take mean spectrum if multiple images were inputted, otherwise plot multiple subplots.
        :param savepath: save path
        :param filename: file name
        :param title: plot title
        """
        embindhist = self.model.calc_embindhist(data, do_return=True)
        if len(self.dendrogram_index) == 0:
            ValueError("No dendrogram_index found. Load dendrogram_index first.")

        embindhist = embindhist[target_vq_layer - 1][
            :, self.dendrogram_index[target_vq_layer - 1]
        ]
        self.plot_feature_spectrum_from_vqindhist(
            embindhist,
            take_mean=take_mean,
            savepath=savepath,
            filename=filename,
            title=title,
        )

    def plot_feature_spectrum_from_vqindhist(
        self,
        embindhist,
        take_mean=True,
        savepath=None,
        filename="Feature_spectrum",
        title=None,
    ):
        """
        Plot feature spectrum from vq index histogram.
        :param embindhist: vq index histogram data; make sure it has 2 dimensions (i.e. batch, index histogram).
        :param take_mean: take mean spectrum if multiple images were inputted, otherwise plot multiple subplots.
        :param savepath: save path
        :param filename: file name
        :param title: plot title
        """

        n_row = 1 if take_mean else embindhist.shape[0]
        if take_mean:
            embindhist = np.mean(embindhist, axis=0, keepdims=True)

        n_index = embindhist.shape[1]
        plt.figure(figsize=(10 * n_index / 136.5, 3 * n_row))
        for i in range(n_row):
            plt.subplot(n_row, 1, i + 1)
            plt.bar(np.arange(n_index), embindhist[i])
            if title:
                if i == 0:
                    plt.title(title)
            plt.ylabel("Counts")
            plt.xlim([0, n_index])
            plt.xticks(np.arange(0, n_index, 100))
        plt.xlabel("Feature index")
        plt.tight_layout()
        if savepath:
            if savepath == "default":
                savepath = self.model.savepath_dict["ft"]
            plt.savefig(os.path.join(savepath, f"{filename}.png"), dpi=300)
        else:
            plt.show()

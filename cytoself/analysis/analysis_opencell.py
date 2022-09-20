import inspect
from os.path import join
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.typing import ArrayLike
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import colorcet as cc

from cytoself.analysis.base import BaseAnalysis
from cytoself.analysis.pearson_correlation import selfpearson_multi


class AnalysisOpenCell(BaseAnalysis):
    """
    Analysis class for OpenCell data
    """

    def __init__(self, datamanager, trainer, homepath: Optional[str] = None, **kwargs):
        super().__init__(datamanager, trainer, homepath, **kwargs)
        self.feature_spectrum_indices = None

    def plot_umap_of_embedding_vector(
        self,
        data_loader: Optional = None,
        label_data: Optional = None,
        umap_data: Optional = None,
        embedding_data: Optional = None,
        image_data: Optional = None,
        group_col: int = 1,
        unique_groups: Optional = None,
        group_annotation: Optional = None,
        savepath_embeddings: Optional[str] = 'default',
        **kwargs,
    ):
        """
        Generate a UMAP plotting from embeddings

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader that will provide both images and labels
        label_data : Numpy array
            Label data (required except when data_loader is used)
        umap_data : Numpy array
            UMAP coordinates (will simply generate a scatter plot from these coordinates)
        embedding_data : Numpy array
            Embedding data (will compute UMAP from the embedding data)
        image_data : Numpy array
            Image data (will compute embeddings, UMAP before generating a scatter plot)
        group_col : int
            The index to be used to group labels
        unique_groups : list or tuple
            The unique groups to be plotted
        group_annotation : Numpy array
            A numpy array that has the same length with the data to be plotted having the group annotation.
        savepath_embeddings : str or None
            The path to save the computed embeddings in the embeddings folder

        """
        if data_loader is None:
            if label_data is None:
                raise ValueError('label_data cannot be None. Provide a 2D-array to label_data.')
        else:
            label_data = data_loader.dataset.label

        # Get compute umap data from embedding_data
        if umap_data is None:
            umap_data = self.compute_umap(data_loader, embedding_data, image_data, savepath_embeddings, **kwargs)

        # Construct group annotation
        label_converted, unique_groups = self.group_labels(label_data, group_col, unique_groups, group_annotation)

        # Making the plot
        scatter_kwargs = {a: kwargs[a] for a in inspect.signature(self.plot_umap_by_group).parameters if a in kwargs}
        self.fig, self.ax = self.plot_umap_by_group(umap_data, label_converted, unique_groups, **scatter_kwargs)

        return umap_data

    def compute_umap(
        self,
        data_loader: Optional = None,
        embedding_data: Optional = None,
        image_data: Optional = None,
        savepath_embeddings: Optional[str] = 'default',
        **kwargs,
    ):
        """
        Compute UMAP

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader that will provide both images and labels
        embedding_data : Numpy array
            Embedding data (will compute UMAP from the embedding data)
        image_data : Numpy array
            Image data (will compute embeddings, UMAP before generating a scatter plot)
        savepath_embeddings : str or None
            The path to save the computed embeddings in the embeddings folder

        Returns
        -------
        Numpy array of UMAP coordinates

        """
        if embedding_data is None:
            print('Computing embeddings from image...')
            embedding_data = self.trainer.infer_embeddings(
                image_data if data_loader is None else data_loader,
                **{a: kwargs[a] for a in inspect.signature(self.trainer.infer_embeddings).parameters if a in kwargs},
            )
            if isinstance(embedding_data, tuple) and len(embedding_data) > 1:
                embedding_data = embedding_data[0]
            if savepath_embeddings is not None:
                if savepath_embeddings == 'default':
                    savepath_embeddings = self.trainer.savepath_dict['embeddings']
                if 'output_layer' in kwargs:
                    fname = kwargs['output_layer']
                else:
                    if 'output_layer' in inspect.signature(self.trainer.infer_embeddings).parameters:
                        fname = inspect.signature(self.trainer.infer_embeddings).parameters['output_layer'].default
                    else:
                        fname = 'embeddings_for_umap'
                np.save(join(savepath_embeddings, fname + '.npy'), embedding_data)
                print(f'embeddings {fname} have been saved at ' + savepath_embeddings)

        print('Computing UMAP coordinates from embeddings...')
        umap_data = self._transform_umap(
            embedding_data,
            **{a: kwargs[a] for a in inspect.signature(self._transform_umap).parameters if a in kwargs},
        )
        return umap_data

    def group_labels(
        self,
        label_data: Optional = None,
        group_col: int = 1,
        unique_groups: Optional = None,
        group_annotation: Optional = None,
    ):
        """
        Generate labels that have group annotations

        Parameters
        ----------
        label_data : Numpy array
            Label data for each data point
        group_col : int
            The index to be used to group labels
        unique_groups : list or tuple
            The unique groups to be plotted
        group_annotation : Numpy array
            A numpy array that has the same length with the data to be plotted having the group annotation.

        Returns
        -------
        A tuple of numpy array

        """
        if unique_groups is None:
            if group_annotation is None:
                unique_groups = np.unique(label_data[:, group_col])
            else:
                unique_groups = np.unique(group_annotation[:, 1])
        label_converted = label_data[:, group_col].astype(object)
        if group_annotation is not None:
            label_converted[:] = 'others'
            for gp in unique_groups:
                label_converted[
                    np.isin(label_data[:, group_col], group_annotation[group_annotation[:, 1] == gp, 0])
                ] = gp
            unique_groups = np.hstack([unique_groups, 'others'])
        return label_converted, unique_groups

    def plot_umap_by_group(
        self,
        umap_data,
        label_data,
        unique_groups: Optional = None,
        colormap: Union[str, tuple] = 'tab20_others',
        s: float = 0.2,
        alpha: float = 0.1,
        title: str = 'UMAP',
        xlabel: str = 'umap1',
        ylabel: str = 'umap2',
        savepath: str = 'default',
        dpi: int = 300,
        figsize: tuple[float, float] = (6, 5),
    ):
        """
        Plot a UMAP by annotating groups in different colors

        Parameters
        ----------
        umap_data : Numpy array
            UMAP coordinates
        label_data : Numpy array
            Label data; has the same length with UMAP data
        unique_groups
        colormap : str or tuple of tuples
            Name of a color map or colormap in RGB or RGBA
            If colormap contains '_others' in its name, the data points labeled with 'others' will be in light gray.
        s : float or int
            Size of data points
        alpha : float or int
            Opacity of data points
        title : str
            Figure title and file name
        xlabel : str
            Label on the x axis
        ylabel : str
            Label on the y axis
        savepath : str
            Path to save the figure
        dpi : int
            Dots per inch
        figsize : tuple of ints or floats
            Figure size

        Returns
        -------
        tuple of figure and axis objects

        """
        if savepath == 'default':
            savepath = join(self.savepath_dict['umap_figures'], title + '.png')
        if isinstance(colormap, str):
            cmap = cm.get_cmap(colormap.replace('_others', '')).colors
        else:
            cmap = colormap
        if unique_groups is None:
            unique_groups = np.unique(label_data)

        fig, ax = plt.subplots(1, figsize=figsize)
        i = 0
        for gp in unique_groups:
            if '_others' in colormap and gp == 'others':
                _c = cm.Greys(25)
            else:
                _c = cmap[i % len(cmap)]
                i += 1
            ind = label_data == gp
            ax.scatter(
                umap_data[ind, 0],
                umap_data[ind, 1],
                s=s,
                alpha=alpha,
                c=np.array(_c).reshape(1, -1),
                label=gp,
                zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        hndls, names = ax.get_legend_handles_labels()
        leg = ax.legend(
            hndls,
            names,
            prop={'size': 6},
            bbox_to_anchor=(1, 1),
            loc='upper left',
            ncol=1 + len(names) // 15,
            frameon=False,
        )
        for ll in leg.legendHandles:
            ll._sizes = [6]
            ll.set_alpha(1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=dpi)
        return fig, ax

    def calculate_cellid_ondim0_vqidx_ondim1(
        self,
        vq_idx: int = 1,
        data_loader: Optional[DataLoader] = None,
        unique_labels: Optional = None,
        label_col: int = 0,
        savepath: Optional[str] = None,
    ):
        """
        Compute the matrix of cell line ID vs. vq index occurrence per image.
        This is needed to compute feature heatmap.

        Parameters
        ----------
        vq_idx : int
            VQ layer index
        data_loader : DataLoader
            DataLoader to compute the matrix
        unique_labels : Sequence or ArrayLike
            Unique labels
        label_col : int
            Label column index to use
        savepath : str
            Path to save the resulting matrix

        Returns
        -------
        Numpy array

        """
        if data_loader is None:
            data_loader = self.datamanager.test_loader
        if unique_labels is None:
            unique_labels = self.datamanager.unique_labels
        indhist = self.trainer.infer_embeddings(data_loader, output_layer=f'vqindhist{vq_idx}')[0]
        print('Computing cell line ID vs vq index...')
        cellid_by_idx = np.zeros((len(unique_labels), indhist.shape[-1]))
        for i, cid in enumerate(tqdm(unique_labels)):
            data0 = indhist[data_loader.dataset.label[:, label_col] == cid]
            cellid_by_idx[i, :] = data0.sum(0) / data0.shape[0]
        if savepath:
            np.save(join(savepath, f'cellid_vqidx{vq_idx}.npy'), cellid_by_idx)
        return cellid_by_idx

    def calculate_corr_vqidx_vqidx(
        self, data: ArrayLike, num_workers: int = 1, filepath: Optional[str] = None
    ) -> ArrayLike:
        """
        Compute self pearson's correlation between vq index and vq index

        Parameters
        ----------
        data : ArrayLike
            Numpy array with VQ index on dim 1
        num_workers : int
            Number of workers
        filepath : str
            File path (including file name & extension)

        Returns
        -------
        Numpy array

        """
        print('Computing self Pearson correlation...')
        corr_idx_idx = np.nan_to_num(selfpearson_multi(data.T, num_workers=num_workers))
        if filepath:
            np.save(filepath, corr_idx_idx)
        return corr_idx_idx

    def plot_clustermap(
        self,
        vq_idx: int = 1,
        data_loader: Optional[DataLoader] = None,
        num_workers: int = 1,
        filepath: str = 'default',
        use_codebook: bool = False,
        update_feature_spectrum_indices: bool = True,
    ):
        """
        Generate hierarchical clustering heatmaps against vqind vs. vqind

        Parameters
        ----------
        vq_idx : int
            VQ layer index
        data_loader : DataLoader
            DataLoader to compute the matrix
        num_workers : int
            Number of workers
        filepath : str
            File path (including file name & extension)
        use_codebook : bool
            Uses codebook to compute self-correlation in VQ indices if Ture, otherwise uses cell id
        update_feature_spectrum_indices : bool
            Overwrite class attribute update_feature_spectrum_indices if True

        Returns
        -------
        Seaborn heatmap object

        """
        if use_codebook:
            _mat_idx = self.trainer.model.vq_layers[vq_idx - 1].codebook.weight.detach().cpu().numpy()
        else:
            _mat_idx = self.calculate_cellid_ondim0_vqidx_ondim1(
                vq_idx=vq_idx, data_loader=data_loader, savepath=self.savepath_dict['feature_spectra_data']
            )
        corr_idx_idx = self.calculate_corr_vqidx_vqidx(
            _mat_idx,
            num_workers=num_workers,
            filepath=join(self.savepath_dict['feature_spectra_data'], f'corr_idx_idx_vq{vq_idx}.npy'),
        )
        print('computing clustermaps...')
        heatmap = sns.clustermap(
            corr_idx_idx,
            cmap=cc.diverging_bwr_20_95_c54,
            vmin=-1,
            vmax=1,
        )
        heatmap.ax_col_dendrogram.set_title(f'vq{vq_idx} indhist Pearson corr hierarchy link')
        heatmap.ax_heatmap.set_xlabel('vq index')
        heatmap.ax_heatmap.set_ylabel('vq index')
        if update_feature_spectrum_indices:
            self.feature_spectrum_indices = np.array(heatmap.dendrogram_row.reordered_ind)

        if filepath:
            if filepath == 'default':
                filepath = join(self.savepath_dict['feature_spectra_figures'], f'clustermap_vq{vq_idx}.png')
            heatmap.savefig(filepath, dpi=300)
        return heatmap

    def compute_feature_spectrum(
        self, vq_index_histogram: ArrayLike, feature_spectrum_indices: Optional[ArrayLike] = None, **kwargs
    ) -> ArrayLike:
        """
        Compute feature spectrum from VQ index histogram

        Parameters
        ----------
        vq_index_histogram : ArrayLike
            2D Numpy array of VQ index histogram
        feature_spectrum_indices : ArrayLike
            1D Numpy array of feature spectrum indices; obtained from cluster map
        kwargs : dict
            kwargs for plot_clustermap method

        Returns
        -------
        2D Numpy array

        """
        if len(vq_index_histogram.shape) != 2:
            raise ValueError('vq_index_histogram must be a 2D matrix.')
        if feature_spectrum_indices is None:
            if self.feature_spectrum_indices is None:
                feature_spectrum_indices = np.array(self.plot_clustermap(**kwargs).dendrogram_row.reordered_ind)
            else:
                feature_spectrum_indices = self.feature_spectrum_indices
        else:
            if len(feature_spectrum_indices.shape) != 1:
                raise ValueError('feature_spectrum_indices must be a 1D array.')
        if vq_index_histogram.shape[-1] != len(feature_spectrum_indices):
            raise ValueError(
                f'The second dim of vq_index_histogram ({vq_index_histogram.shape[-1]}) '
                f'must be same as the length of feature_spectrum_indices ({len(feature_spectrum_indices)}).'
            )

        return vq_index_histogram[:, feature_spectrum_indices]

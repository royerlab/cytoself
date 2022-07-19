import inspect
from os.path import join
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from cytoself.analysis.base import BaseAnalysis


class AnalysisOpenCell(BaseAnalysis):
    """
    Analysis class for OpenCell data
    """

    def __init__(self, datamanager, trainer, homepath: Optional[str] = None, **kwargs):
        super().__init__(datamanager, trainer, homepath, **kwargs)

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
        **kwargs,
    ):
        # Get compute umap data from embedding_data
        if umap_data is None:
            if embedding_data is None:
                print('Computing embeddings from image...')
                if data_loader is None:
                    if label_data is None:
                        raise ValueError('label_data cannot be None. Provide a 2D-array to label_data.')
                else:
                    label_data = data_loader.dataset.label
                embedding_data = self.trainer.infer_embeddings(
                    image_data if data_loader is None else data_loader,
                    **{a: kwargs[a] for a in inspect.getfullargspec(self.trainer.infer_embeddings).args if a in kwargs},
                )
                if isinstance(embedding_data, tuple) and len(embedding_data) > 1:
                    embedding_data = embedding_data[0]
            print('Computing UMAP coordinates from embeddings...')
            umap_data = self._transform_umap(
                embedding_data,
                **{a: kwargs[a] for a in inspect.getfullargspec(self._transform_umap).args if a in kwargs},
            )

        if label_data is None:
            raise ValueError('label_data cannot be None. Provide a 2D-array to label_data.')

        # Construct group annotation
        if unique_groups is None:
            if group_annotation is None:
                unique_groups = np.unique(label_data[:, group_col])
            else:
                unique_groups = np.unique(group_annotation[:, 1])
        label_converted = label_data[:, group_col].astype(object)
        label_converted[:] = 'others'
        if group_annotation is not None:
            for gp in unique_groups:
                label_converted[np.isin(label_data, group_annotation[group_annotation[:, 1] == gp, 0])[:, 0]] = gp

        # Making the plot
        scatter_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(self.plot_umap_by_group).args if a in kwargs}
        self.fig, self.ax = self.plot_umap_by_group(umap_data, label_converted, unique_groups, **scatter_kwargs)

        return umap_data

    def plot_umap_by_group(
        self,
        umap_data,
        label_data,
        unique_groups: Optional = None,
        colormap: str = 'tab20',
        s: Union[float, int] = 0.2,
        alpha: Union[float, int] = 0.1,
        title: str = 'UMAP',
        xlabel: str = 'umap1',
        ylabel: str = 'umap2',
        savepath: str = 'default',
        dpi: int = 300,
        figsize: tuple[Union[float, int], Union[float, int]] = (6, 5),
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
        colormap : str
            Name of a color map
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
        cmap = cm.get_cmap(colormap).colors
        if unique_groups is None:
            unique_groups = np.unique(label_data)

        fig, ax = plt.subplots(1, figsize=figsize)
        for i, gp in enumerate(unique_groups):
            ind = label_data == gp
            ax.scatter(
                umap_data[ind, 0],
                umap_data[ind, 1],
                s=s,
                alpha=alpha,
                c=np.array(cmap[i % len(cmap)]).reshape(1, -1),
                label=gp,
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        hndls, names = ax.get_legend_handles_labels()
        leg = ax.legend(
            hndls, names, prop={'size': 6}, bbox_to_anchor=(1, 1), loc='upper left', ncol=1 + len(names) // 15
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

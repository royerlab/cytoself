import inspect
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        group_col: int = 0,
        unique_groups: Optional = None,
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

        # Configure arguments
        local_figsize = kwargs['figsize'] if 'figsize' in kwargs else (6.4, 4.8)
        scatter_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(plt.scatter).args if a in kwargs}
        if unique_groups is None:
            unique_groups = np.unique(label_data[:, group_col])

        # Making the plot
        self.fig, self.ax = plt.subplots(figsize=local_figsize)
        for gp in tqdm(unique_groups):
            ind = label_data[:, group_col] == gp
            self.ax.scatter(umap_data[ind, 0], umap_data[ind, 1], label=gp, **scatter_kwargs)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        if 'title' in kwargs:
            self.ax.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            self.ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            self.ax.set_ylabel(kwargs['ylabel'])
        if 'show_legend' in kwargs and kwargs['show_legend']:
            leg = self.ax.legend(frameon=False)
            for lh in leg.legendHandles:
                lh._sizes = [plt.rcParams['font.size'] * 2.4]
                lh.set_alpha(1)
        self.fig.tight_layout()

        if 'savepath' in kwargs:
            self.fig.savefig(kwargs['savepath'], dpi=kwargs['dpi'] if 'dpi' in kwargs else 'figure')

        return umap_data

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
                    if image_data is None:
                        raise ValueError('image_data is required.')
                    if label_data is None:
                        raise ValueError('label_data is required.')
                embedding_data = self.trainer.infer_embeddings(
                    image_data if data_loader is None else data_loader,
                    **{a: kwargs[a] for a in inspect.getfullargspec(self.trainer.infer_embeddings).args if a in kwargs},
                )
                if data_loader is not None:
                    embedding_data, label_data = embedding_data
            print('Computing UMAP coordinates from embeddings...')
            umap_data = self._transform_umap(
                embedding_data,
                **{a: kwargs[a] for a in inspect.getfullargspec(self._transform_umap).args if a in kwargs},
            )

        # Configure arguments
        local_figsize = kwargs['figsize'] if 'figsize' in kwargs else (6.4, 4.8)
        scatter_kwargs = {a: kwargs[a] for a in inspect.getfullargspec(plt.scatter).args if a in kwargs}
        if unique_groups is None:
            unique_groups = np.unique(label_data[:, group_col])

        # Making the plot
        fig, ax = plt.subplots(figsize=local_figsize)
        for gp in tqdm(unique_groups):
            ind = label_data[:, group_col] == gp
            ax.scatter(umap_data[ind, 0], umap_data[ind, 1], label=gp, **scatter_kwargs)
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        fig.tight_layout()

        if 'savepath' in kwargs:
            fig.savefig(kwargs['savepath'], dpi=kwargs['dpi'] if 'dpi' in kwargs else 'figure')

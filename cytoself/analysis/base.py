import os.path
from os.path import join
from typing import Optional

import umap


class BaseAnalysis:
    """
    Base class for Analysis
    """

    def __init__(
        self,
        datamanager,
        trainer,
        homepath: Optional[str] = None,
        **kwargs,
    ):
        self.datamanager = datamanager
        self.trainer = trainer
        self.reducer = None
        self.savepath_dict = {
            'homepath': join(trainer.savepath_dict['homepath'], 'analysis') if homepath is None else homepath
        }
        self._init_savepath()
        self.fig = None
        self.ax = None

    def _init_savepath(self):
        folders = ['umap_figures', 'umap_data', 'feature_spectra_figures', 'feature_spectra_data']
        for f in folders:
            p = join(self.savepath_dict['homepath'], f)
            if not os.path.exists(p):
                os.makedirs(p)
            self.savepath_dict[f] = p

    def _fit_umap(self, data, n_neighbors=15, min_dist=0.1, metric='euclidean', verbose=True, **kwargs):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, verbose=verbose, **kwargs)
        self.reducer.fit(data.reshape(data.shape[0], -1))

    def _transform_umap(self, data, n_neighbors=15, min_dist=0.1, metric='euclidean', verbose=True, **kwargs):
        if self.reducer is None:
            self._fit_umap(data, n_neighbors, min_dist, metric, verbose, **kwargs)
        try:
            return self.reducer.transform(data.reshape(data.shape[0], -1))
        except Exception as e:
            raise ValueError(
                'Error at reducer.transform \n' + str(e),
                '\n\nThe input data dimension may be incompatible with pre-computed UMAP.'
                'Try to reset the pre-computed UMAP by running Analysis.reset_umap().',
            )

    def reset_umap(self):
        self.reducer = None

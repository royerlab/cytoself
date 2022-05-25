from os.path import exists, join
import numpy as np
from ..base import BaseAnalysis
from cytoself.trainer.test.test_vanilla_trainer import test_VanillaAETrainer


class test_BaseAnalysis(test_VanillaAETrainer):
    def setUp(self):
        super().setUp()
        self.analysis = BaseAnalysis(self.datamgr, self.trainer)

    def test__init_savepath(self):
        self.analysis._init_savepath()
        for f in ['umap_figures', 'umap_data', 'feature_spectra_figures', 'feature_spectra_data']:
            assert f in self.analysis.savepath_dict
            assert exists(join(self.analysis.savepath_dict['homepath'], f))

    def test__fit_umap(self):
        self.analysis._fit_umap(np.random.randn(100, 10), verbose=False)
        assert self.analysis.reducer.embedding_.shape == (100, 2)

    def test__transform_umap(self):
        output = self.analysis._transform_umap(np.random.randn(100, 10))
        assert (output == self.analysis.reducer.embedding_).all()

        self.analysis.reducer = None
        output = self.analysis._transform_umap(np.random.randn(100, 10))
        assert (output == self.analysis.reducer.embedding_).all()

from os.path import exists, join

import numpy as np
import pytest

from cytoself.analysis.base import BaseAnalysis


@pytest.fixture(scope='module')
def base_analysis(vanilla_ae_trainer, opencell_datamgr_vanilla):
    return BaseAnalysis(opencell_datamgr_vanilla, vanilla_ae_trainer)


def test_base_analysis__init_savepath(base_analysis):
    base_analysis._init_savepath()
    for f in ['umap_figures', 'umap_data', 'feature_spectra_figures', 'feature_spectra_data']:
        assert f in base_analysis.savepath_dict
        assert exists(join(base_analysis.savepath_dict['homepath'], f))


def test_base_analysis__fit_umap(base_analysis):
    base_analysis._fit_umap(np.random.randn(100, 10), verbose=False)
    assert base_analysis.reducer.embedding_.shape == (100, 2)


def test_base_analysis__transform_umap(base_analysis):
    inputs = np.random.randn(100, 10)
    base_analysis.reset_umap()
    output = base_analysis._transform_umap(inputs)
    assert (output == base_analysis.reducer.embedding_).all()
    output2 = base_analysis._transform_umap(inputs)
    assert (output2 == base_analysis.reducer.embedding_).all()
    with pytest.raises(ValueError):
        base_analysis._transform_umap(np.random.randn(100, 20))

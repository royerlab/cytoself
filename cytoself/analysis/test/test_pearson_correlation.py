import numpy as np
from cytoself.analysis.pearson_correlation import corr_single, selfpearson_multi, pearson_multi


def test_corr_single():
    corr = corr_single(2, np.arange(5), np.tile(np.arange(5), (5, 1)), 7)
    assert np.unique(corr[:, :2]) == 0
    assert np.unique(corr[:, 2:]) == 1


def test_selfpearson_multi():
    corr = selfpearson_multi(np.tile(np.arange(5), (3, 1)))
    assert np.unique(corr) == 1


def test_pearson_multi():
    corr = pearson_multi(np.arange(5), np.tile(np.arange(5), (5, 1)))
    assert np.unique(corr) == 1

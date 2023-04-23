import numpy as np
import pytest

from cytoself.analysis.utils.pearson_correlation import corr_single, pearson_multi, selfpearson_multi


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
    with pytest.raises(ValueError):
        pearson_multi(np.arange(5), np.tile(np.arange(4), (5, 1)))

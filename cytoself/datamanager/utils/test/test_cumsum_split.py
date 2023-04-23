import numpy as np

from cytoself.datamanager.utils.cumsum_split import cumsum_split

counts = np.random.randint(10, 50, size=20)


def test_cumsum_split_idx():
    splits = (0.8, 0.1, 0.1)
    out = cumsum_split(counts, splits)
    sums = [sum(counts[i]) for i in out]
    for i, d in enumerate(splits):
        assert d * 0.7 < sums[i] / sum(sums) < d * 1.4

    splits = (8, 1, 1)
    out = cumsum_split(counts, splits)
    sums = [sum(counts[i]) for i in out]
    for i, d in enumerate(splits):
        assert d * 0.7 * 0.1 < sums[i] / sum(sums) < d * 1.4 * 0.1


def test_cumsum_split_arr():
    splits = (0.8, 0.1, 0.1)
    out = cumsum_split(counts, splits, np.arange(len(counts)))
    sums = [sum(counts[i]) for i in out]
    for i, d in enumerate(splits):
        assert d * 0.7 < sums[i] / sum(sums) < d * 1.4

    splits = (8, 1, 1)
    out = cumsum_split(counts, (8, 1, 1), np.arange(len(counts)))
    sums = [sum(counts[i]) for i in out]
    for i, d in enumerate(splits):
        assert d * 0.75 * 0.1 < sums[i] / sum(sums) < d * 1.32 * 0.1

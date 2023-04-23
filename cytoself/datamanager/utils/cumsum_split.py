from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


def cumsum_split(counts: Sequence, split_perc: tuple, arr: Optional[ArrayLike] = None):
    """
    Splits an array by the cumulative sum.

    Parameters
    ----------
    counts : numpy array
        A numpy array of cell counts in each FOV.
    split_perc : tuple
        The split percentages
    arr : numpy array
        The numpy array to be split

    Returns
    -------
    Numpy arrays of indices if arr is None, otherwise the resulting split arr.

    """
    if arr is not None:
        assert len(counts) == len(arr)
    if sum(split_perc) != 1:
        split_perc = [i / sum(split_perc) for i in split_perc]

    # Sort counts in descendent order
    ind0 = np.argsort(counts)[::-1]
    counts = counts[ind0]

    # Find split indices
    cumsum_counts = np.cumsum(counts)
    count_limits = np.cumsum(np.array(split_perc)[:-1] * cumsum_counts[-1])
    split_counter = 0
    split_indices = []
    for i, csum in enumerate(cumsum_counts):
        if csum >= count_limits[split_counter]:
            split_counter += 1
            split_indices.append(i)
        if split_counter == len(count_limits):
            break
    rng0 = [0] + split_indices
    rng1 = split_indices + [len(counts)]
    ind = [ind0[i0:i1] for i0, i1 in zip(rng0, rng1)]

    if arr is None:
        return ind
    else:
        return [arr[d] for d in ind]

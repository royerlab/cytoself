from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from cytoself.datamanager.utils.cumsum_split import cumsum_split


def single_proc(label: Sequence, split_perc: tuple, fovpath_idx: int):
    """
    Single-processing unit for data splitting on FOVs

    Parameters
    ----------
    label : numpy array
        The label data in numpy array
    split_perc : tuple
        The split percentages
    fovpath_idx : int
        The index of label data to determine FOVs

    Returns
    -------
    A list of boolean array to split data according to the data split ratio.

    """
    uniq, counts = np.unique(label[:, fovpath_idx], return_counts=True)
    fovpaths = cumsum_split(counts, split_perc, uniq)
    return [np.isin(label[:, fovpath_idx], pths) for pths in fovpaths]


def splitdata_on_fov(
    label_all: ArrayLike,
    split_perc: tuple,
    cellline_id_idx: int,
    fovpath_idx: int,
    num_workers: int = 4,
    shuffle_seed: int = 1,
):
    """
    A multiprocessing function to split data on FOVs

    Parameters
    ----------
    label_all : numpy array
        The label data in numpy array
    split_perc : tuple
        The split percentages
    cellline_id_idx : int
        The index of label data to determine proteins
    fovpath_idx : int
        The index of label data to determine FOVs
    num_workers : int
        The number of workers for the multiprocessing
    shuffle_seed : int
        A random seed for shuffling

    Returns
    -------
    A tuple of indices.

    """
    df_label_all = pd.DataFrame(label_all)
    df_label_all_gp = df_label_all.groupby(cellline_id_idx)
    cell_line_id = df_label_all_gp.count().index.to_numpy()

    results = Parallel(n_jobs=num_workers)(
        delayed(single_proc)(label_all[label_all[:, cellline_id_idx] == cid], split_perc, fovpath_idx)
        for cid in tqdm(cell_line_id)
    )

    train_ind, val_ind, test_ind = [], [], []
    for d, cid in zip(results, cell_line_id):
        idx0 = np.where(label_all[:, cellline_id_idx] == cid)[0]
        train_ind.append(idx0[d[0]])
        val_ind.append(idx0[d[1]])
        test_ind.append(idx0[d[2]])
    if len(train_ind) > 0:
        train_ind = np.hstack(train_ind)
    if len(val_ind) > 0:
        val_ind = np.hstack(val_ind)
    if len(test_ind) > 0:
        test_ind = np.hstack(test_ind)

    if shuffle_seed:
        np.random.seed(shuffle_seed)
        np.random.shuffle(train_ind)
        np.random.shuffle(val_ind)
        np.random.shuffle(test_ind)

    return train_ind, val_ind, test_ind

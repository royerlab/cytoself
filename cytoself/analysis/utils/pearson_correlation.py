import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.stats import pearsonr
from tqdm import tqdm


def corr_single(offset: int, array: ArrayLike, matrix: ArrayLike, dims: int) -> ArrayLike:
    """
    Compute pearson's correlation between an array & a matrix

    Parameters
    ----------
    offset : int
        Offset
    array : ArrayLike
        1D Numpy array
    matrix : ArrayLike
        Numpy array
    dims : int
        Shape size in the second dimension of the correlation

    Returns
    -------
    Numpy array

    """
    corr = np.zeros((1, dims))
    for ii, row in enumerate(matrix):
        corr[:, ii + offset] = pearsonr(array, row)[0]
    return corr


def selfpearson_multi(data: ArrayLike, num_workers: int = 1) -> ArrayLike:
    """
    Compute self pearson correlation using multiprocessing

    Parameters
    ----------
    data : ArrayLike
        2D Numpy array; self-correlation is performed on axis 1 and iterates over axis 0
    num_workers : int
        Number of workers

    Returns
    -------
    2D Numpy array

    """
    corr = Parallel(n_jobs=num_workers, prefer='threads')(
        delayed(corr_single)(i, row, data[i:], data.shape[0]) for i, row in enumerate(tqdm(data))
    )
    corr = np.vstack(corr)
    corr_up = np.triu(corr, k=1)
    return corr_up.T + corr


def pearson_multi(array: ArrayLike, matrix: ArrayLike, num_workers: int = 1) -> ArrayLike:
    """
    Compute pearson's correlation between an array & a matrix

    Parameters
    ----------
    array : ArrayLike
        1D Numpy array
    matrix : ArrayLike
        2D Numpy matrix
    num_workers : int
        Number of workers

    Returns
    -------
    Numpy array

    """
    if array.shape != matrix.shape[1:]:
        raise ValueError('array.shape must equal matrix.shape[1:].')
    corr = Parallel(n_jobs=num_workers)(delayed(pearsonr)(array, ar) for ar in tqdm(matrix))
    return np.vstack(corr)[:, 0]

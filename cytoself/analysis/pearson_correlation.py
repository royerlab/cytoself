import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from tqdm import tqdm


def corr_single(i1, ar1, dim, data1):
    """
    Compute pearson's correlation with an arrays vs. a matrix;
    :param i1: index of iterations
    :param ar1: target array
    :param dim: number of arrays to compute correlation
    :param data1: target matrix
    :return: an array of correlation
    """
    corr = np.zeros((1, dim))
    for i2, ar2 in enumerate(data1):
        corr[:, i2 + i1] = pearsonr(np.nan_to_num(ar1), np.nan_to_num(ar2))[0]
    return corr


def selfpearson_multi(data, num_cores=10, axis=-1):
    """
    Compute self pearson correlation with parallel computing
    :param data: 2D matrix
    :param num_cores: number of cores for computation
    :param axis: self-correlation along axis
    :return: correlation 2D matrix
    """
    if axis == -1:
        data = data.T
    corr = Parallel(n_jobs=num_cores, prefer="threads")(
        delayed(corr_single)(i1, ar1, data.shape[0], data[i1:])
        for i1, ar1 in enumerate(tqdm(data))
    )
    corr = np.vstack(corr)
    corr_up = np.triu(corr, k=1)
    return corr_up.T + corr


def pearson_multi_1toM(target, matrix, num_cores=10):
    """
    Compute pearson's correlation with 1 vs. a matrix
    :param target: target array
    :param matrix: comparison matrix
    :return: a correlation array
    """
    if target.shape != matrix.shape[1:]:
        raise ValueError(
            "The shape of target must be the same as the shape[1:] of the matrix."
        )
    corr = Parallel(n_jobs=num_cores)(
        delayed(pearsonr)(target, ar) for ar in tqdm(matrix)
    )
    return np.vstack(corr)[:, 0]

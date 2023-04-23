from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


def calculate_cluster_centrosize(data: ArrayLike, cluster_label: ArrayLike, exclude: Optional[Sequence] = None):
    """
    Generates a matrix with different clusters on dimension 0
    and cluster name, centroid coordinates & cluster size on dimension 1.
    It will be used to compute the cluster score for a given UMAP

    Parameters
    ----------
    data : ArrayLike
        UMAP data
    cluster_label : ArrayLike
        Numpy array labeling each cluster; must be same length as data
    exclude : str or Sequence
        Labels to exclude from calculating cluster sizes

    Returns
    -------
    Numpy array with cluster_name, centroid, cluster_size on each column

    """
    cluster_uniq = np.unique(cluster_label)
    if exclude is not None:
        cluster_uniq = cluster_uniq[~np.isin(cluster_uniq, exclude)]

    centroid_list = []
    clustersize_list = []
    for cl in cluster_uniq:
        ind = cluster_label == cl
        data0 = data[ind.flatten()]
        centroid = np.median(data0, axis=0)
        # square distance between each datapoint and centroid
        square_distance = (centroid - data0) ** 2
        # median of sqrt of square_distance as cluster size
        cluster_size = np.median(np.sqrt(square_distance[:, 0] + square_distance[:, 1]))
        centroid_list.append(centroid)
        clustersize_list.append(cluster_size)
    return np.vstack([cluster_uniq, np.vstack(centroid_list).T, clustersize_list]).T

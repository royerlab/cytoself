from os.path import join

import numpy as np

from cytoself.analysis.plot_umap import plot_umap_with_cluster_score
from cytoself.analysis.test.test_analysis_opencell import assert_not_raises


def test_plot_umap_with_cluster_score(basepath):
    label = np.random.choice(2, (10,)).astype(object)
    label[label == 0] = 'others'
    label[label == 1] = 'group1'
    with assert_not_raises():
        plot_umap_with_cluster_score(
            np.random.random((10, 2)),
            label,
            show_size_on_legend=True,
            show_cluster_area=True,
        )
        plot_umap_with_cluster_score(
            np.random.random((10, 2)),
            label,
            colormap='tab20',
            show_size_on_legend=True,
            show_cluster_area=True,
            filepath=join(basepath, 'umap_with_cluster_score.png'),
        )

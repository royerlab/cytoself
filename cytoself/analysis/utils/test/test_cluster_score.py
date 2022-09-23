import numpy as np

from cytoself.analysis.utils.cluster_score import calculate_cluster_centrosize


def test_calculate_cluster_centrosize():
    data = np.random.random((100, 2))
    cluster_label = np.random.randint(6, size=(100,)).astype(str)
    output = calculate_cluster_centrosize(data, cluster_label, ['5'])
    assert output.shape == (5, 4)

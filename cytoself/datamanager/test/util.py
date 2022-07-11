from torch.utils.data import Dataset

from cytoself.datamanager.preloaded_dataset import PreloadedDataset


def assert_instance(dataset, data_len):
    assert len(dataset) == data_len
    assert isinstance(dataset, Dataset)


def assert_dataset(dataset, keys=('image', 'label')):
    assert isinstance(dataset, PreloadedDataset)
    d = next(iter(dataset))
    assert isinstance(d, dict)
    if 'image' in keys:
        assert d['image'].shape == (2, 10, 10)
    if 'label' in keys:
        assert len(d['label']) == 3

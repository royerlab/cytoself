from torch.utils.data import Dataset

from cytoself.datamanager.preloaded_dataset import PreloadedDataset


def assert_instance(dataset, data_len):
    assert len(dataset) == data_len
    assert isinstance(dataset, Dataset)


def assert_dataset(dataset, label_len=1):
    assert isinstance(dataset, PreloadedDataset)
    current_dataset = next(iter(dataset))
    assert isinstance(current_dataset, dict)
    if 'image' in current_dataset:
        assert current_dataset['image'].shape == (2, 10, 10)
    if 'label' in current_dataset:
        if len(current_dataset['label'].shape) > 0:
            assert len(current_dataset['label']) == label_len

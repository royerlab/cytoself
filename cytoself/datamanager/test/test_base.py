import numpy as np
import pytest
from cytoself.datamanager.base import DataManagerBase
from cytoself.datamanager.preloaded_dataset import PreloadedDataset
from .util import assert_instance
from ..utils.test.test_splitdata_on_fov import gen_label

test_label = gen_label()
data_len = len(test_label)
test_data = np.zeros((data_len, 2, 100, 100), dtype=np.uint8)


def test_label_only():
    dataset = PreloadedDataset(test_label)
    assert_instance(dataset, data_len)
    d = next(iter(dataset))
    assert isinstance(d, dict)
    for key, val in d.items():
        assert key == 'label'
        assert isinstance(val, np.ndarray)
        assert len(val) == 1


def test_label_converter():
    with pytest.raises(ValueError):
        dataset = PreloadedDataset(test_label, label_format='onehot')
        dataset.label_converter(test_label[:10])
    uniq = np.unique(test_data[:, 0])
    dataset = PreloadedDataset(test_label, unique_labels=uniq, label_format='onehot')
    assert (dataset.label_converter(test_label[:10]) == (test_label[:10, None] == uniq)).all()
    dataset = PreloadedDataset(test_label, unique_labels=uniq, label_format='index')
    assert (dataset.label_converter(test_label[:10]) == (test_label[:10, None] == uniq).argmax(1)).all()


def test_label_and_image():
    dataset = PreloadedDataset(test_label, test_data)
    assert_instance(dataset, data_len)
    d = next(iter(dataset))
    assert isinstance(d, dict)
    assert len(d) == 2
    for key, val in d.items():
        if key == 'label':
            assert isinstance(val, np.ndarray)
            assert len(val) == 1
        else:
            assert key == 'image'
            assert val.shape == (2, 100, 100)


def test_transform():
    dataset = PreloadedDataset(test_label, test_data, lambda x: x + 1)
    assert_instance(dataset, data_len)
    d = next(iter(dataset))
    assert isinstance(d, dict)
    assert len(d) == 2
    for key, val in d.items():
        if key == 'label':
            assert isinstance(val, np.ndarray)
            assert len(val) == 1
        else:
            assert key == 'image'
            assert (val == test_data + 1).all()


def test_DataManagerBase():
    basepath = 'basepath'
    data_split = (0.8, 0.1, 0.1)
    batch_size = 32
    shuffle_seed = 1
    num_workers = 4
    datamgr = DataManagerBase(basepath, data_split, batch_size, shuffle_seed, num_workers)
    assert datamgr.basepath == basepath
    assert datamgr.data_split == data_split
    assert datamgr.batch_size == batch_size
    assert datamgr.shuffle_seed == shuffle_seed
    assert datamgr.num_workers == num_workers
    assert hasattr(datamgr, 'train_dataset')
    assert hasattr(datamgr, 'val_dataset')
    assert hasattr(datamgr, 'test_dataset')
    assert hasattr(datamgr, 'train_loader')
    assert hasattr(datamgr, 'val_loader')
    assert hasattr(datamgr, 'test_loader')
    assert datamgr.const_dataset() is None

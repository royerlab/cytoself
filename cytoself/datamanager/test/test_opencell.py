import tempfile
from shutil import rmtree
from math import floor, ceil
from os.path import join
import numpy as np
import pytest
import torch

from .util import assert_dataset
from ..utils.test.test_splitdata_on_fov import gen_label
from ..opencell import DataManagerOpenCell, get_file_df

n_fovs = 40
test_label = gen_label(n_fovs)


@pytest.fixture(scope="session")
def basepath():
    basepath = tempfile.mkdtemp()

    shape = (2, 10, 10)

    for i in range(test_label[:, 0].max() + 1):
        ind = test_label[:, 0] == i
        np.save(join(basepath, f'protein{i}_nuc.npy'), np.zeros((sum(ind),) + shape, dtype=np.uint8))
        np.save(join(basepath, f'protein{i}_label.npy'), test_label[ind])

    yield basepath

    rmtree(basepath)


@pytest.fixture
def opencell_datamgr(basepath):
    datamgr = DataManagerOpenCell(basepath, ['nuc'])
    assert 'label' in datamgr.channel_list
    assert 'nuc' in datamgr.channel_list

    return datamgr


def test_typeerror(basepath):
    with pytest.raises(TypeError):
        get_file_df(basepath, 1)


def test_label_only(basepath):
    df = get_file_df(basepath)
    for i, row in df.iterrows():
        assert df['label'].str.contains(join(basepath, f'protein{i}_label.npy')).any()


def test_label_image(basepath):
    df = get_file_df(basepath, ('label', 'nuc'))
    for i, row in df.iterrows():
        assert df['label'].str.contains(join(basepath, f'protein{i}_label.npy')).any()
        assert df['nuc'].str.contains(join(basepath, f'protein{i}_nuc.npy')).any()

# TODO: test loading order


def test_intensity_adjustment(basepath):
    datamgr = DataManagerOpenCell(
        basepath, ['nuc'], intensity_adjustment={'gfp': 20, 'nuc': 5, 'nucdist': 0.02}
    )
    assert datamgr.intensity_adjustment['gfp'] == 20
    assert datamgr.intensity_adjustment['nuc'] == 5
    assert datamgr.intensity_adjustment['nucdist'] == 0.02


def test_determine_load_paths(basepath, opencell_datamgr):
    df_toload = opencell_datamgr.determine_load_paths(
        labels_toload=['protein0', 'protein1', 'protein2'],
        labels_tohold=['protein0'],
    )
    assert df_toload['label'].str.contains(join(basepath, 'protein1_label.npy')).any()


def test__load_data_multi(opencell_datamgr):
    df_toload = opencell_datamgr.determine_load_paths(suffix=opencell_datamgr.channel_list)
    image_all, label_all = opencell_datamgr._load_data_multi(df_toload)
    assert len(image_all) == len(test_label)
    assert len(label_all) == len(test_label)
    assert np.isin(label_all[:, 1], [f'fov{i}' for i in range(n_fovs)]).all()


def test_split_data(opencell_datamgr):
    df_toload = opencell_datamgr.determine_load_paths(suffix=opencell_datamgr.channel_list)
    image_all, label_all = opencell_datamgr._load_data_multi(df_toload)
    index_list = opencell_datamgr.split_data(label_all)
    for d, s in zip(index_list, opencell_datamgr.data_split):
        data = test_label[d]
        assert (
                min(1, floor(len(label_all) * s * 0.6)) <= len(data) <= ceil(len(label_all) * s * 1.5)
        ), 'Split ratio deviates too far.'


def test_split_data_notfov(basepath, opencell_datamgr):
    datamgr = DataManagerOpenCell(basepath, ['label'], fov_col=None)
    df_toload = datamgr.determine_load_paths(suffix=opencell_datamgr.channel_list)
    image_all, label_all = datamgr._load_data_multi(df_toload)
    index_list = datamgr.split_data(label_all)
    for d, s in zip(index_list, opencell_datamgr.data_split):
        data = test_label[d]
        assert (
                min(1, floor(len(label_all) * s * 0.7)) <= len(data) <= ceil(len(label_all) * s * 1.4)
        ), 'Split ratio deviates too far.'


def test_const_label_book(opencell_datamgr):
    opencell_datamgr.const_label_book(test_label)
    assert (np.unique(test_label[:, 0]) == opencell_datamgr.unique_labels).all()


def test_const_dataset(opencell_datamgr):
    opencell_datamgr.const_dataset()
    assert_dataset(opencell_datamgr.train_dataset)
    assert_dataset(opencell_datamgr.val_dataset)
    assert_dataset(opencell_datamgr.test_dataset)


def test_const_dataset_labelonly(basepath):
    datamgr = DataManagerOpenCell(basepath, ['label'])
    datamgr.const_dataset()
    assert_dataset(datamgr.train_dataset, keys=('labels',))
    assert_dataset(datamgr.val_dataset, keys=('labels',))
    assert_dataset(datamgr.test_dataset, keys=('labels',))


def test_const_dataloader(opencell_datamgr):
    opencell_datamgr.const_dataset(label_format=None)
    with pytest.raises(TypeError):
        opencell_datamgr.const_dataloader(shuffle=False)

    opencell_datamgr.const_dataset()
    opencell_datamgr.const_dataloader(shuffle=False)
    train_data0 = next(iter(opencell_datamgr.train_dataset))
    train_batch = next(iter(opencell_datamgr.train_loader))
    assert len(train_batch['label']) == opencell_datamgr.batch_size
    assert torch.is_tensor(train_batch['label'])
    assert torch.is_tensor(train_batch['image'])
    assert tuple(train_batch['label'].shape)[1:] == train_data0['label'].shape
    assert tuple(train_batch['image'].shape)[1:] == train_data0['image'].shape

import os
from math import ceil, floor
from os.path import exists, join

import numpy as np
import pytest
import torch

from cytoself.datamanager.opencell import DataManagerOpenCell, get_file_df
from cytoself.datamanager.test.util import assert_dataset
from cytoself.test_util.test_parameters import n_fovs, test_label


def test_typeerror(gen_data_2x10x10):
    with pytest.raises(TypeError):
        get_file_df(gen_data_2x10x10, 1)


def test_label_only(gen_data_2x10x10):
    df = get_file_df(gen_data_2x10x10)
    for i, row in df.iterrows():
        assert df['label'].str.contains(join(gen_data_2x10x10, f'protein{i}_label.npy')).any()


def test_label_image(gen_data_2x10x10):
    df = get_file_df(gen_data_2x10x10, ('label', 'nuc'))
    for i, row in df.iterrows():
        assert df['label'].str.contains(join(gen_data_2x10x10, f'protein{i}_label.npy')).any()
        assert df['nuc'].str.contains(join(gen_data_2x10x10, f'protein{i}_nuc.npy')).any()


# TODO: test loading order


def test_intensity_adjustment(gen_data_2x10x10):
    datamgr = DataManagerOpenCell(
        gen_data_2x10x10, ['nuc'], intensity_adjustment={'pro': 20, 'nuc': 5, 'nucdist': 0.02}
    )
    assert datamgr.intensity_adjustment['pro'] == 20
    assert datamgr.intensity_adjustment['nuc'] == 5
    assert datamgr.intensity_adjustment['nucdist'] == 0.02


def test_determine_load_paths(gen_data_2x10x10, opencell_datamgr_2x10x10):
    df_toload = opencell_datamgr_2x10x10.determine_load_paths(
        labels_toload=['protein0', 'protein1', 'protein2'],
        labels_tohold=['protein0'],
    )
    assert df_toload['label'].str.contains(join(gen_data_2x10x10, 'protein1_label.npy')).any()


def test__load_data_multi(opencell_datamgr_2x10x10):
    opencell_datamgr_2x10x10.intensity_adjustment = {'nuc': 1}
    df_toload = opencell_datamgr_2x10x10.determine_load_paths(suffix=opencell_datamgr_2x10x10.channel_list)
    image_all, label_all = opencell_datamgr_2x10x10._load_data_multi(df_toload)
    assert len(image_all) == len(test_label)
    assert len(label_all) == len(test_label)
    assert np.isin(label_all[:, 1], [f'fov{i}' for i in range(n_fovs)]).all()


def test_split_data(opencell_datamgr_2x10x10):
    df_toload = opencell_datamgr_2x10x10.determine_load_paths(suffix=opencell_datamgr_2x10x10.channel_list)
    image_all, label_all = opencell_datamgr_2x10x10._load_data_multi(df_toload)
    index_list = opencell_datamgr_2x10x10.split_data(label_all)
    for d, s in zip(index_list, opencell_datamgr_2x10x10.data_split):
        data = test_label[d]
        assert (
            min(1, floor(len(label_all) * s * 0.6)) <= len(data) <= ceil(len(label_all) * s * 1.5)
        ), 'Split ratio deviates too far.'


def test_split_data_notfov(gen_data_2x10x10, opencell_datamgr_2x10x10):
    datamgr = DataManagerOpenCell(gen_data_2x10x10, ['label'], fov_col=None)
    df_toload = datamgr.determine_load_paths(suffix=opencell_datamgr_2x10x10.channel_list)
    image_all, label_all = datamgr._load_data_multi(df_toload)
    index_list = datamgr.split_data(label_all)
    for d, s in zip(index_list, opencell_datamgr_2x10x10.data_split):
        data = test_label[d]
        assert (
            min(1, floor(len(label_all) * s * 0.7)) <= len(data) <= ceil(len(label_all) * s * 1.4)
        ), 'Split ratio deviates too far.'


def test_const_label_book(opencell_datamgr_2x10x10):
    opencell_datamgr_2x10x10.const_label_book(test_label)
    assert (np.unique(test_label[:, 0]) == opencell_datamgr_2x10x10.unique_labels).all()


def test_const_dataset(opencell_datamgr_2x10x10):
    opencell_datamgr_2x10x10.const_dataloader()
    assert_dataset(opencell_datamgr_2x10x10.train_loader.dataset)
    assert_dataset(opencell_datamgr_2x10x10.val_loader.dataset)
    assert_dataset(opencell_datamgr_2x10x10.test_loader.dataset)
    opencell_datamgr_2x10x10.const_dataloader(label_format='onehot')
    assert_dataset(opencell_datamgr_2x10x10.train_loader.dataset, label_len=3)
    assert_dataset(opencell_datamgr_2x10x10.val_loader.dataset, label_len=3)
    assert_dataset(opencell_datamgr_2x10x10.test_loader.dataset, label_len=3)


def test_const_dataset_labelonly(gen_data_2x10x10):
    datamgr = DataManagerOpenCell(gen_data_2x10x10, ['label'])
    datamgr.const_dataloader()
    assert_dataset(datamgr.train_loader.dataset)
    assert_dataset(datamgr.val_loader.dataset)
    assert_dataset(datamgr.test_loader.dataset)


def test_const_dataloader(opencell_datamgr_2x10x10):
    with pytest.raises(TypeError):
        opencell_datamgr_2x10x10.const_dataloader(label_format=None, shuffle=False)

    opencell_datamgr_2x10x10.const_dataloader(shuffle=False)
    train_batch = next(iter(opencell_datamgr_2x10x10.train_loader))
    assert len(train_batch['label']) == opencell_datamgr_2x10x10.train_loader.batch_size
    assert torch.is_tensor(train_batch['label'])
    assert torch.is_tensor(train_batch['image'])
    assert train_batch['label'].shape == train_batch['image'].shape[:1]


@pytest.mark.slow
def test_download_sample_data(opencell_datamgr_2x10x10):
    dest = join(opencell_datamgr_2x10x10.basepath, 'example_data')
    opencell_datamgr_2x10x10.download_sample_data(output=dest)
    assert exists(dest)
    assert len(os.listdir(dest)) == 36

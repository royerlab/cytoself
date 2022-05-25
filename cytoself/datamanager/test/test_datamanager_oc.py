import tempfile
from unittest import TestCase
from shutil import rmtree
from math import floor, ceil
from os.path import join
import numpy as np
import torch

from ..base import PreloadedDataset
from ..utils.test.test_splitdata_on_fov import gen_label
from ..datamanager_oc import DataManagerOpenCell, get_file_df

n_fovs = 40
test_label = gen_label(n_fovs)


class TmpDirTestCase(TestCase):
    def setUp(self):
        self._basepath = tempfile.mkdtemp()

    def tearDown(self):
        rmtree(self._basepath)

    def gen_npy(self, shape=(2, 10, 10)):
        for i in range(test_label[:, 0].max() + 1):
            ind = test_label[:, 0] == i
            np.save(join(self._basepath, f'protein{i}_nuc.npy'), np.zeros((sum(ind),) + shape, dtype=np.uint8))
            np.save(join(self._basepath, f'protein{i}_label.npy'), test_label[ind])


class test_get_file_df(TmpDirTestCase):
    def setUp(self, shape=(2, 10, 10)):
        super().setUp()
        self.gen_npy(shape)

    def test_typeerror(self):
        with self.assertRaises(TypeError):
            get_file_df(self._basepath, 1)

    def test_label_only(self):
        df = get_file_df(self._basepath)
        for i, row in df.iterrows():
            assert df['label'].str.contains(join(self._basepath, f'protein{i}_label.npy')).any()

    def test_label_image(self):
        df = get_file_df(self._basepath, ('label', 'nuc'))
        for i, row in df.iterrows():
            assert df['label'].str.contains(join(self._basepath, f'protein{i}_label.npy')).any()
            assert df['nuc'].str.contains(join(self._basepath, f'protein{i}_nuc.npy')).any()


class test_DataManagerOpenCell(test_get_file_df):
    def setUp(self):
        super().setUp()
        self.datamgr = DataManagerOpenCell(self._basepath, ['nuc'])
        assert 'label' in self.datamgr.channel_list
        assert 'nuc' in self.datamgr.channel_list

    def _assert_dataset(self, dataset, keys=('image', 'label')):
        assert isinstance(dataset, PreloadedDataset)
        for d in dataset:
            assert isinstance(d, dict)
            if 'image' in keys:
                assert d['image'].shape == (2, 10, 10)
            if 'label' in keys:
                assert len(d['label']) == 2
            break

    def test_intensity_adjustment(self):
        datamgr = DataManagerOpenCell(
            self._basepath, ['nuc'], intensity_adjustment={'gfp': 20, 'nuc': 5, 'nucdist': 0.02}
        )
        assert datamgr.intensity_adjustment['gfp'] == 20
        assert datamgr.intensity_adjustment['nuc'] == 5
        assert datamgr.intensity_adjustment['nucdist'] == 0.02

    def test_determine_load_paths(self):
        df_toload = self.datamgr.determine_load_paths(
            labels_toload=['protein0', 'protein1', 'protein2'],
            labels_tohold=['protein0'],
        )
        assert df_toload['label'].str.contains(join(self._basepath, 'protein1_label.npy')).any()

    def test__load_data_multi(self):
        df_toload = self.datamgr.determine_load_paths(suffix=self.datamgr.channel_list)
        image_all, label_all = self.datamgr._load_data_multi(df_toload)
        assert len(image_all) == len(test_label)
        assert len(label_all) == len(test_label)
        assert np.isin(label_all[:, 1], [f'fov{i}' for i in range(n_fovs)]).all()

    def test_split_data(self):
        df_toload = self.datamgr.determine_load_paths(suffix=self.datamgr.channel_list)
        image_all, label_all = self.datamgr._load_data_multi(df_toload)
        index_list = self.datamgr.split_data(label_all)
        for d, s in zip(index_list, self.datamgr.data_split):
            data = test_label[d]
            assert (
                min(1, floor(len(label_all) * s * 0.66)) <= len(data) <= ceil(len(label_all) * s * 1.42)
            ), 'Split ratio deviates too far.'

    def test_split_data_notfov(self):
        datamgr = DataManagerOpenCell(self._basepath, ['label'], fov_col=None)
        df_toload = datamgr.determine_load_paths(suffix=self.datamgr.channel_list)
        image_all, label_all = datamgr._load_data_multi(df_toload)
        index_list = datamgr.split_data(label_all)
        for d, s in zip(index_list, self.datamgr.data_split):
            data = test_label[d]
            assert (
                min(1, floor(len(label_all) * s * 0.7)) <= len(data) <= ceil(len(label_all) * s * 1.4)
            ), 'Split ratio deviates too far.'

    def test_const_label_book(self):
        self.datamgr.const_label_book(test_label)
        assert (np.unique(test_label[:, 0]) == self.datamgr.unique_labels).all()

    def test_const_dataset(self):
        self.datamgr.const_dataset()
        self._assert_dataset(self.datamgr.train_dataset)
        self._assert_dataset(self.datamgr.val_dataset)
        self._assert_dataset(self.datamgr.test_dataset)

    def test_const_dataset_labelonly(self):
        datamgr = DataManagerOpenCell(self._basepath, ['label'])
        datamgr.const_dataset()
        self._assert_dataset(datamgr.train_dataset, keys=('labels',))
        self._assert_dataset(datamgr.val_dataset, keys=('labels',))
        self._assert_dataset(datamgr.test_dataset, keys=('labels',))

    def test_const_dataloader(self):
        self.datamgr.const_dataset(label_format=None)
        with self.assertRaises(TypeError):
            self.datamgr.const_dataloader(shuffle=False)

        self.datamgr.const_dataset()
        self.datamgr.const_dataloader(shuffle=False)
        for train_data0 in self.datamgr.train_dataset:
            break
        train_batch = next(iter(self.datamgr.train_loader))
        assert len(train_batch['label']) == self.datamgr.batch_size
        assert torch.is_tensor(train_batch['label'])
        assert torch.is_tensor(train_batch['image'])
        assert tuple(train_batch['label'].shape)[1:] == train_data0['label'].shape
        assert tuple(train_batch['image'].shape)[1:] == train_data0['image'].shape

    def tearDown(self):
        super().tearDown()

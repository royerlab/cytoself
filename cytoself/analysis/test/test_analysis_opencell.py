import os
from os.path import exists, join
from contextlib import contextmanager
import pytest

import numpy as np
from ..analysis_opencell import AnalysisOpenCell
from cytoself.trainer.test.test_vanilla_trainer import setup_VanillaAETrainer


@contextmanager
def assert_not_raises():
    try:
        yield
    except Exception as e:
        raise pytest.fail(f'Did raise {e}')


class test_BaseAnalysis(setup_VanillaAETrainer):
    def setUp(self):
        super().setUp()
        self.analysis = AnalysisOpenCell(self.datamgr, self.trainer)
        self.label_data = np.repeat(np.arange(10), 10).reshape(-1, 1)
        np.random.shuffle(self.label_data)
        self.file_name = join(self.analysis.savepath_dict['umap_figures'], 'test.png')

    def test_plot_umap_of_embedding_vector_nullinput(self):
        with self.assertRaises(ValueError):
            self.analysis.plot_umap_of_embedding_vector(label_data=self.label_data)
        with self.assertRaises(ValueError):
            self.analysis.plot_umap_of_embedding_vector(image_data=np.random.randn(100, 1, 32, 32))

    def test_plot_umap_of_embedding_vector_umapdata(self):
        with assert_not_raises():
            self.analysis.plot_umap_of_embedding_vector(
                label_data=self.label_data,
                umap_data=np.random.randn(100, 2),
                savepath=self.file_name,
                title='fig title',
                xlabel='x axis',
                ylabel='y axis',
            )
        assert exists(self.file_name)
        os.remove(self.file_name)

    def test_plot_umap_of_embedding_vector_embedding(self):
        with assert_not_raises():
            self.analysis.plot_umap_of_embedding_vector(
                label_data=self.label_data,
                embedding_data=np.random.randn(100, 10),
                savepath=self.file_name,
                title='fig title',
                xlabel='x axis',
                ylabel='y axis',
            )
        assert exists(self.file_name)
        os.remove(self.file_name)

    def test_plot_umap_of_embedding_vector_image(self):
        with assert_not_raises():
            self.analysis.plot_umap_of_embedding_vector(
                label_data=self.datamgr.test_loader.dataset.label,
                image_data=self.datamgr.test_loader.dataset.data,
                savepath=self.file_name,
                title='fig title',
                xlabel='x axis',
                ylabel='y axis',
            )
        assert exists(self.file_name)
        os.remove(self.file_name)

    def test_plot_umap_of_embedding_vector_dataloader(self):
        self.datamgr.const_dataset(label_format='index')
        self.datamgr.const_dataloader()
        with assert_not_raises():
            self.analysis.plot_umap_of_embedding_vector(
                data_loader=self.datamgr.test_loader,
                savepath=self.file_name,
                title='fig title',
                xlabel='x axis',
                ylabel='y axis',
            )
        assert exists(self.file_name)
        os.remove(self.file_name)

import os
from os.path import exists, join
from contextlib import contextmanager
import pytest
import matplotlib.pyplot as plt
import numpy as np
from cytoself.analysis.analysis_opencell import AnalysisOpenCell


@contextmanager
def assert_not_raises():
    try:
        yield
    except Exception as e:
        raise pytest.fail(f'Did raise {e}')


@pytest.fixture(scope='module')
def analysis_opencell(vanilla_ae_trainer, opencell_datamgr_vanilla):
    return AnalysisOpenCell(opencell_datamgr_vanilla, vanilla_ae_trainer)


@pytest.fixture(scope='module')
def _file_name(analysis_opencell):
    return join(analysis_opencell.savepath_dict['umap_figures'], 'test.png')


@pytest.fixture(scope='module')
def _label_data():
    label_data = np.repeat(np.arange(10), 10).reshape(-1, 1)
    np.random.shuffle(label_data)
    return label_data


def test_plot_umap_of_embedding_vector_nullinput(analysis_opencell, _label_data):
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(label_data=_label_data)
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(image_data=np.random.randn(100, 1, 32, 32))
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(embedding_data=np.random.randn(100, 10))


def test_plot_umap_of_embedding_vector_umapdata(analysis_opencell, _file_name, _label_data):
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            label_data=_label_data,
            umap_data=np.random.randn(100, 2),
            group_col=0,
            savepath=_file_name,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            show_legend=True,
            figsize=(6, 5),
            dpi=100,
        )
    assert exists(_file_name)
    assert output.shape == (100, 2)
    assert isinstance(analysis_opencell.fig, type(plt.figure()))
    os.remove(_file_name)


def test_plot_umap_of_embedding_vector_embedding(analysis_opencell, _file_name, _label_data):
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            label_data=_label_data,
            embedding_data=np.random.randn(100, 10),
            group_col=0,
            savepath=_file_name,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            show_legend=True,
        )
    assert exists(_file_name)
    assert output.shape == (100, 2)
    os.remove(_file_name)


def test_plot_umap_of_embedding_vector_image(analysis_opencell, _file_name, opencell_datamgr_vanilla):
    analysis_opencell.reset_umap()
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            label_data=opencell_datamgr_vanilla.test_loader.dataset.label,
            image_data=opencell_datamgr_vanilla.test_loader.dataset.data,
            savepath=_file_name,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            show_legend=True,
        )
    assert exists(_file_name)
    assert output.shape == (len(opencell_datamgr_vanilla.test_loader.dataset.label), 2)
    os.remove(_file_name)


def test_plot_umap_of_embedding_vector_dataloader(analysis_opencell, _file_name, opencell_datamgr_vanilla):
    analysis_opencell.reset_umap()
    opencell_datamgr_vanilla.const_dataset(label_format='index')
    opencell_datamgr_vanilla.const_dataloader()
    group_annotation = np.tile(np.arange(3).reshape(-1, 1), (1, 2)).astype(object)
    group_annotation[:2, 1] = 'gp0'
    group_annotation[2:, 1] = 'gp1'
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            data_loader=opencell_datamgr_vanilla.test_loader,
            savepath=_file_name,
            group_annotation=group_annotation,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            show_legend=True,
        )
    assert exists(_file_name)
    assert output.shape == (len(opencell_datamgr_vanilla.test_loader.dataset.label), 2)
    os.remove(_file_name)

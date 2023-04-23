import inspect
from contextlib import contextmanager
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
def analysis_cytoselflite(cytoselflite_trainer, opencell_datamgr_vanilla):
    return AnalysisOpenCell(opencell_datamgr_vanilla, cytoselflite_trainer)


@pytest.fixture(scope='module')
def _file_name(analysis_opencell):
    return join(analysis_opencell.savepath_dict['umap_figures'], 'test.png')


@pytest.fixture(scope='module')
def _label_data():
    label_data = np.repeat(np.arange(10), 10).reshape(-1, 1)
    np.random.shuffle(label_data)
    return label_data


def test_compute_umap(analysis_cytoselflite, opencell_datamgr_vanilla):
    analysis_cytoselflite.compute_umap(opencell_datamgr_vanilla.test_loader)
    fname = inspect.signature(analysis_cytoselflite.trainer.infer_embeddings).parameters['output_layer'].default
    assert exists(join(analysis_cytoselflite.trainer.savepath_dict['embeddings'], fname + '.npy'))
    analysis_cytoselflite.reset_umap()
    analysis_cytoselflite.compute_umap(opencell_datamgr_vanilla.test_loader, output_layer='vqindhist2')
    assert exists(join(analysis_cytoselflite.trainer.savepath_dict['embeddings'], 'vqindhist2.npy'))


def test_group_labels(analysis_opencell, opencell_datamgr_vanilla):
    _lab = opencell_datamgr_vanilla.test_loader.dataset.label
    output = analysis_opencell.group_labels(_lab, group_col=0)
    assert (output[0] == opencell_datamgr_vanilla.test_loader.dataset.label[:, 0]).all()
    assert (output[1] == np.arange(3)).all()
    group_annotation = np.array([[0, 'gp0'], [1, 'gp1']], dtype=object)
    output = analysis_opencell.group_labels(_lab, group_annotation=group_annotation, group_col=0)
    assert np.unique(output[0][_lab[:, 0] == 0]) == 'gp0'
    assert np.unique(output[0][_lab[:, 0] == 1]) == 'gp1'
    assert np.unique(output[0][_lab[:, 0] == 2]) == 'others'
    assert (output[1] == np.array(['gp0', 'gp1', 'others'])).all()


def test_plot_umap_of_embedding_vector_nullinput(analysis_opencell, _label_data):
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(label_data=_label_data)
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(image_data=np.random.randn(100, 1, 32, 32))
    with pytest.raises(ValueError):
        analysis_opencell.plot_umap_of_embedding_vector(embedding_data=np.random.randn(100, 10))


def test_plot_umap_by_group(analysis_opencell):
    label = np.random.choice(2, (10,)).astype(object)
    label[label == 0] = 'others'
    label[label == 1] = 'group1'
    with assert_not_raises():
        analysis_opencell.plot_umap_by_group(np.random.random((10, 2)), label)


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
            figsize=(6, 5),
            dpi=100,
        )
    assert exists(_file_name)
    assert output.shape == (100, 2)
    assert isinstance(analysis_opencell.fig, type(plt.figure()))


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
            colormap='tab20',
        )
    assert exists(_file_name)
    assert output.shape == (100, 2)


def test_plot_umap_of_embedding_vector_image(analysis_opencell, _file_name, opencell_datamgr_vanilla):
    analysis_opencell.reset_umap()
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            label_data=opencell_datamgr_vanilla.test_loader.dataset.label,
            image_data=opencell_datamgr_vanilla.test_loader.dataset.data,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            colormap='tab10',
        )
    assert exists(join(analysis_opencell.savepath_dict['umap_figures'], 'fig title.png'))
    assert output.shape == (len(opencell_datamgr_vanilla.test_loader.dataset.label), 2)


def test_plot_umap_of_embedding_vector_dataloader(analysis_opencell, _file_name, opencell_datamgr_vanilla):
    analysis_opencell.reset_umap()
    opencell_datamgr_vanilla.const_dataloader(label_format='index')
    group_annotation = np.array([[0, 'gp0'], [1, 'gp1']], dtype=object)
    with assert_not_raises():
        output = analysis_opencell.plot_umap_of_embedding_vector(
            data_loader=opencell_datamgr_vanilla.test_loader,
            savepath=_file_name,
            group_annotation=group_annotation,
            title='fig title',
            xlabel='x axis',
            ylabel='y axis',
            colormap=((0.5, 0.1, 0.2, 0.7), (0.2, 0.6, 0.4, 0.5)),
        )
    assert exists(_file_name)
    assert exists(join(join(analysis_opencell.trainer.savepath_dict['embeddings'], 'embeddings_for_umap.npy')))
    assert output.shape == (len(opencell_datamgr_vanilla.test_loader.dataset.label), 2)


def test_calc_cellid_vqidx(analysis_cytoselflite):
    cellid_by_idx = analysis_cytoselflite.calculate_cellid_ondim0_vqidx_ondim1(
        savepath=analysis_cytoselflite.savepath_dict['feature_spectra_data']
    )
    assert cellid_by_idx.shape == (3, 7)
    assert exists(join(analysis_cytoselflite.savepath_dict['feature_spectra_data'], 'cellid_vqidx1.npy'))


def test_calc_corr_idx_idx(analysis_cytoselflite):
    corr = analysis_cytoselflite.calculate_cellid_ondim0_vqidx_ondim1()
    corr_idx_idx = analysis_cytoselflite.calculate_corr_vqidx_vqidx(
        corr, filepath=join(analysis_cytoselflite.savepath_dict['feature_spectra_data'], 'corr_idx_idx.npy')
    )
    assert corr_idx_idx.shape == (7, 7)
    assert exists(join(analysis_cytoselflite.savepath_dict['feature_spectra_data'], 'corr_idx_idx.npy'))


def test_plot_clustermap(analysis_cytoselflite):
    heatmap = analysis_cytoselflite.plot_clustermap(use_codebook=True)
    assert hasattr(heatmap, 'dendrogram_row')

    heatmap = analysis_cytoselflite.plot_clustermap()
    assert hasattr(heatmap, 'dendrogram_row')
    assert exists(join(analysis_cytoselflite.savepath_dict['feature_spectra_figures'], 'clustermap_vq1.png'))


def test_compute_feature_spectrum(analysis_cytoselflite):
    with pytest.raises(ValueError):
        analysis_cytoselflite.compute_feature_spectrum(np.random.randint(7, size=(7,)))
    with pytest.raises(ValueError):
        analysis_cytoselflite.compute_feature_spectrum(
            np.random.randint(7, size=(1, 7)), np.random.choice(7, size=(5,))
        )
    with pytest.raises(ValueError):
        analysis_cytoselflite.compute_feature_spectrum(
            np.random.randint(7, size=(1, 7)), np.random.choice(7, size=(1, 7))
        )
    analysis_cytoselflite.feature_spectrum_indices = None
    vq_indhist = np.random.randint(7, size=(4, 7))
    ft_spec = analysis_cytoselflite.compute_feature_spectrum(vq_indhist)
    assert ft_spec.shape == (4, 7)
    ft_spec2 = analysis_cytoselflite.compute_feature_spectrum(vq_indhist)
    assert (ft_spec == ft_spec2).all()

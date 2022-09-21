import tempfile
from os.path import join
from os import makedirs
from shutil import rmtree

import pytest

from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.test_util.dummy_data_generation import gen_npy
from cytoself.test_util.test_parameters import add_default_model_args, CYTOSELF_MODEL_ARGS
from cytoself.trainer.cytoselflite_trainer import CytoselfLiteTrainer
from cytoself.trainer.cytoselffull_trainer import CytoselfFullTrainer
from cytoself.trainer.vanilla_trainer import VanillaAETrainer


@pytest.fixture(scope='session')
def basepath():
    basepath = tempfile.mkdtemp()

    yield basepath

    rmtree(basepath)


@pytest.fixture(scope='session')
def gen_data_2x10x10(basepath):
    path = join(basepath, "2x10x10")
    makedirs(path)
    gen_npy(path, (10, 10))
    return path


@pytest.fixture(scope='session')
def gen_data_1x32x32(basepath):
    path = join(basepath, "1x32x32")
    makedirs(path)
    gen_npy(path, (32, 32))
    return path


@pytest.fixture(scope='module')
def opencell_datamgr_2x10x10(gen_data_2x10x10):
    datamgr = DataManagerOpenCell(gen_data_2x10x10, ['pro', 'nuc'])
    assert 'label' in datamgr.channel_list
    assert 'nuc' in datamgr.channel_list
    assert 'pro' in datamgr.channel_list
    return datamgr


@pytest.fixture(scope='module')
def opencell_datamgr_1x32x32(gen_data_1x32x32):
    datamgr = DataManagerOpenCell(gen_data_1x32x32, ['nuc'])
    assert 'label' in datamgr.channel_list
    assert 'nuc' in datamgr.channel_list
    return datamgr


@pytest.fixture(scope='module')
def opencell_datamgr_vanilla(gen_data_1x32x32):
    datamgr = DataManagerOpenCell(gen_data_1x32x32, ['nuc'])
    datamgr.const_dataloader(batch_size=2)
    return datamgr


@pytest.fixture(scope='module')
def vanilla_ae_trainer(basepath):
    model_args = {
        'input_shape': (1, 32, 32),
        'emb_shape': (16, 16, 16),
        'output_shape': (1, 32, 32),
    }
    model_args = add_default_model_args(model_args)
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    return VanillaAETrainer(train_args, homepath=basepath, model_args=model_args)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_heavy = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_heavy)


@pytest.fixture(scope='module')
def cytoselflite_trainer(basepath):
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    return CytoselfLiteTrainer(train_args, homepath=basepath, model_args=CYTOSELF_MODEL_ARGS)


@pytest.fixture(scope='module')
def cytoselffull_trainer(basepath):
    train_args = {'lr': 1e-6, 'max_epoch': 2}
    return CytoselfFullTrainer(train_args, homepath=basepath, model_args=CYTOSELF_MODEL_ARGS)

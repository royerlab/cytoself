from os.path import exists, join

import numpy as np
import pandas as pd
import pytest

from cytoself.trainer.utils.plot_history import plot_history, plot_history_cytoself


@pytest.fixture(scope='module')
def generate_history():
    df = pd.DataFrame(
        {
            n: np.arange(10) + np.random.randn(10)
            for n in [
                'train_loss',
                'val_loss',
                'test_loss',
                'train_fc_loss',
                'val_fc_loss',
                'lr',
                'train_perplexity1',
                'train_perplexity2',
            ]
        }
    )
    df['test_loss'][:-1] = None
    return df


def test_plot_history(generate_history, basepath):
    fpath = join(basepath, 'history.png')
    plot_history(
        generate_history,
        metrics1=['train_loss', 'val_loss', 'test_loss', 'train_fc_loss'],
        metrics2='lr',
        title='training history',
        ylabel2='learn rate',
        savepath=basepath,
        file_name='history.png',
    )
    assert exists(fpath)

    fpath = join(basepath, 'history2.png')
    with pytest.warns():
        plot_history(
            generate_history,
            metrics1='test_fc_loss',
            metrics2='lr',
            title='training history',
            ylabel2='fc loss',
            savepath=basepath,
            file_name='history2.png',
        )
    assert exists(fpath)


def test_plot_history_cytoself(generate_history, basepath):
    fpath = join(basepath, 'training_history.png')
    plot_history_cytoself(
        generate_history,
        savepath=basepath,
        file_name='training_history.png',
    )
    assert exists(fpath)

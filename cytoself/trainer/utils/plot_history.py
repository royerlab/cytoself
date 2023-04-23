from os.path import join
from typing import Optional, Sequence, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


def _draw_history_axis(axis: plt.Axes, history: pd.DataFrame, metrics: str, cmap: Sequence):
    """
    Generate the axis object for a history plot

    Parameters
    ----------
    axis : matplotlib.pyplot.axes.Axes
        Axes object to draw history
    history : DataFrame
        A DataFrame of training history
    metrics : str
        Metrics name
    cmap : A Sequence of tuple or list
        A Sequence of tuple or list as a color map
    Returns
    -------
    lines : list
        A list of Line2D object
    axis : matplotlib.pyplot.axes.Axes
        Axes object with history drawn on it

    """
    lines = []
    color_dict = {}
    counts = 0
    for metric_name in metrics:
        if metric_name in history:
            _y_val = history[metric_name]
            if 'train' in metric_name:
                linestyle, marker = '-', ''
                metric_name_shared = metric_name.replace('train', '')
            elif 'val' in metric_name:
                linestyle, marker = '--', ''
                metric_name_shared = metric_name.replace('val', '')
            elif 'test' in metric_name:
                linestyle, marker = ':', ''
                metric_name_shared = metric_name.replace('test', '')
            else:
                linestyle, marker = '', '.'
                metric_name_shared = metric_name
            if _y_val.notna().sum() < 2:
                marker = 'o'
            if metric_name_shared not in color_dict:
                color_dict[metric_name_shared] = cmap[counts % 10]
                counts += 1

            if min(_y_val) > 0:
                lines += axis.semilogy(_y_val, linestyle + marker, c=color_dict[metric_name_shared], label=metric_name)
            else:
                lines += axis.plot(_y_val, linestyle + marker, c=color_dict[metric_name_shared], label=metric_name)
        else:
            warn(metric_name + ' cannot be found in history.', UserWarning)
    return lines, axis


def plot_history(
    history: pd.DataFrame,
    ax1: Optional[plt.axis] = None,
    metrics1: Union[str, Sequence[str]] = 'loss',
    metrics2: Optional[Union[str, Sequence[str]]] = None,
    title: Optional[str] = None,
    xlabel: str = 'Epoch',
    ylabel1: Optional[str] = 'Loss',
    ylabel2: Optional[str] = None,
    legend_fontsize: float = 8,
    savepath: Optional[str] = None,
    file_name: str = 'history.png',
    dpi: int = 300,
):
    """

    Parameters
    ----------
    history : pandas.DataFrame
        A DataFrame of training history
    ax1 : matplotlib.pyplot.axis
        Axis object to integrate; A new Axis object will be created if None.
    metrics1 : str
        Metrics to plot on the left y-axis
    metrics2 : str
        Metrics to plot on the right y-axis
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel1 : str
        Left y-axis label
    ylabel2 : str
        Right y-axis label
    legend_fontsize : float
        Legend font size
    savepath : str
        Path to save plot
    file_name : str
        File name
    dpi : int
        Dots per inch

    """

    if ax1 is None:
        _, ax1 = plt.subplots()
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel1:
        ax1.set_ylabel(ylabel1)
    if isinstance(metrics1, str):
        metrics1 = [metrics1]
    lines1, ax1 = _draw_history_axis(ax1, history, metrics1, cm.tab10.colors)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if metrics2:
        ax2 = ax1.twinx()
        if ylabel2:
            ax2.set_ylabel(ylabel2)
        if isinstance(metrics2, str):
            metrics2 = [metrics2]
        lines2, ax2 = _draw_history_axis(ax2, history, metrics2, cm.tab20.colors[1::2])
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        lines1 += lines2

    labs = [ln.get_label() for ln in lines1]
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(lines1, labs, prop={'size': legend_fontsize})
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(join(savepath, file_name), dpi=dpi)


def plot_history_cytoself(
    history,
    title: Optional[str] = 'training history',
    savepath: Optional[str] = None,
    file_name: str = 'training_history.png',
    dpi: int = 300,
):
    """
    Generates a multi-panel plot of training history perticularly for a cytoself model

    Parameters
    ----------
    history : DataFrame
        DataFrame of training history
    title : str
        Title of the plot
    savepath : str
        Saving path
    file_name : str
        File name
    dpi : int
        Dots per inch of the plot

    """
    n_row, n_col = 2, 3
    col = history.columns.str.replace(r'\d+', '', regex=True)
    metrics1_list = [
        {'Overall': ['train_loss', 'val_loss', 'test_loss']},
        {'Reconstruction': history.columns[col.str.contains('reconstruction')]},
        {'Classification': history.columns[col.str.contains('fc_')]},
        {'Vector Quantization': history.columns[col.str.contains('vq_')]},
        {'Perplexity': history.columns[col.str.contains('perplexity')]},
    ]

    fig, ax = plt.subplots(n_row, n_col, figsize=(7 * n_col, 5.5 * n_row))
    counts = 0
    for i in range(n_row * n_col):
        ix, iy = np.unravel_index(i, (n_row, n_col))
        if i == 0 and title is not None:
            ax[ix, iy].text(
                0.02,
                0.98,
                title,
                fontsize=18,
                verticalalignment='top',
            )
            ax[ix, iy].axis('off')
        else:
            (key,), (val,) = zip(*metrics1_list[counts].items())
            plot_history(
                history,
                ax[ix, iy],
                metrics1=val,
                metrics2='lr',
                title=key,
                ylabel1=key if key == 'Perplexity' else 'Loss',
                ylabel2='Learn rate',
                savepath=None,
            )
            counts += 1
    fig.tight_layout()
    if savepath is not None:
        plt.savefig(join(savepath, file_name), dpi=dpi)

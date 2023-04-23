from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from cytoself.analysis.utils.cluster_score import calculate_cluster_centrosize


def plot_umap_with_cluster_score(
    umap_data,
    label_data,
    unique_groups: Optional = None,
    colormap: Union[str, tuple] = 'tab20_others',
    s: float = 0.2,
    alpha: float = 0.1,
    title: str = 'UMAP',
    xlabel: str = 'umap1',
    ylabel: str = 'umap2',
    filepath: str = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (6, 5.5),
    show_size_on_legend: bool = True,
    show_cluster_area: bool = True,
    centroid_kwargs: Optional[dict] = None,
    circle_kwargs: Optional[dict] = None,
    text_kwargs: Optional[dict] = None,
):
    """
    Plot a UMAP with cluster scores annotated

    Parameters
    ----------
    umap_data : Numpy array
        UMAP coordinates
    label_data : Numpy array
        Label data; has the same length with UMAP data
    unique_groups
    colormap : str or tuple of tuples
        Name of a color map or colormap in RGB or RGBA
        If colormap contains '_others' in its name, the data points labeled with 'others' will be in light gray.
    s : float or int
        Size of data points
    alpha : float or int
        Opacity of data points
    title : str
        Figure title and file name
    xlabel : str
        Label on the x axis
    ylabel : str
        Label on the y axis
    filepath : str
        Path to save the figure
    dpi : int
        Dots per inch
    figsize : tuple of ints or floats
        Figure size
    show_size_on_legend : bool
        Show cluster size of each group in legend if True
    show_cluster_area : bool
        Show cluster area in the scatter plot if True
    centroid_kwargs : dict
        kwargs for drawing cluster centroid in the scatter plot
    circle_kwargs : dict
        kwargs for drawing cluster area in the scatter plot
    text_kwargs : dict
        kwargs for drawing cluster size in the scatter plot

    Returns
    -------
    tuple of figure and axis objects

    """
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    if unique_groups is None:
        unique_groups = np.unique(label_data)
    if centroid_kwargs is None:
        centroid_kwargs = {}
    centroid_kwargs = {
        k: centroid_kwargs[k] if k in centroid_kwargs else v
        for k, v in {'marker': 'x', 's': 1, 'c': 'red', 'lw': 0.2}.items()
    }
    if circle_kwargs is None:
        circle_kwargs = {}
    circle_kwargs = {
        k: circle_kwargs[k] if k in circle_kwargs else v
        for k, v in {'ls': '--', 'lw': 0.2, 'ec': 'red', 'fill': False}.items()
    }
    if text_kwargs is None:
        text_kwargs = {}
    text_kwargs = {k: text_kwargs[k] if k in text_kwargs else v for k, v in {'fontsize': 6, 'c': 'red'}.items()}

    cluster_matrix = calculate_cluster_centrosize(umap_data, label_data, 'others' if '_others' in colormap else None)
    intra_cluster = np.median(cluster_matrix[:, -1].astype(float))
    inter_cluster = np.std(cluster_matrix[:, 1:-1].astype(float))
    cluster_score = inter_cluster / intra_cluster

    fig, ax = plt.subplots(1, figsize=figsize)
    i = 0
    for gp in unique_groups:
        ind = label_data == gp
        legend_label = gp

        # Compute cluster matrix
        cluster_row = cluster_matrix[cluster_matrix[:, 0] == gp].ravel()
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(25)
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        if show_size_on_legend and gp != 'others':
            legend_label += f' {cluster_row[-1]:0.2e}'

        # Draw a scatter plot
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=legend_label,
            zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
        )

        if show_cluster_area and gp != 'others':
            ax.scatter(cluster_row[1], cluster_row[2], **centroid_kwargs)
            circle = plt.Circle(cluster_row[1:3], cluster_row[-1], **circle_kwargs)
            ax.add_artist(circle)
            ax.text(
                cluster_row[1] + cluster_row[-1] / 2,
                cluster_row[2] + cluster_row[-1] / 2,
                f'{cluster_row[-1]:.2f}',
                **text_kwargs,
            )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hndls, names = ax.get_legend_handles_labels()
    if 'others' in names:
        ind = names.index('others')
        names[ind:-1], names[-1] = names[ind + 1 :], names[ind]
        hndls[ind:-1], hndls[-1] = hndls[ind + 1 :], hndls[ind]
    leg = ax.legend(
        hndls,
        names,
        prop={'size': 6},
        bbox_to_anchor=(1, 1),
        loc='upper left',
        ncol=1 + len(names) // 20,
        frameon=False,
    )
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + f'\ncluster score {cluster_score}')
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=dpi)
    return fig, ax

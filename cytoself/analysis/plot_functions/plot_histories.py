import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from warnings import warn


def plothistory(
    hist,
    ax1=None,
    metrics1="loss",
    metrics2="lr",
    metrics2_label="learn rate",
    title=None,
    xlabel="Epoch",
    ylabel1="Loss",
    ylabel2="Learn rate",
    legend_fontsize=8,
    savepath=None,
    dpi=300,
):
    """
    :param hist: history object or dictionary
    :param ax1: axis object for plotting in a subplot
    :param metrics1: metrics to be plotted on the left y axis
    :param metrics2: metrics to be plotted on the right y axis
    :param metrics2_label: legend label for metrics 2
    :param title: Title for the plot
    :param xlabel: xlabel
    :param ylabel1: ylabel on the left axis
    :param ylabel2: ylabel on the right axis
    :param legend_fontsize: font size for legend
    :param savepath: save path
    :param dpi: dpi for the saved plot
    """
    if ax1 is None:
        _, ax1 = plt.subplots()
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel1:
        ax1.set_ylabel(ylabel1)
    if not isinstance(metrics1, list):
        metrics1 = [metrics1]
    if not isinstance(metrics2, list):
        metrics2 = [metrics2]
    if not isinstance(metrics2_label, list):
        metrics2_label = [metrics2_label]
    if len(metrics2) != len(metrics2_label):
        metrics2_label = metrics2
        warn(
            "The length of metrics2 is inconsistent with that of metrics2_label.\n"
            "metrics2 will be used as metrics2_label."
        )

    numepo = len(hist["epoch"])
    ax2 = ax1.twinx()
    lns = []
    count = 0
    lns1 = []
    for m in metrics1:
        color = None
        for key, value in hist.items():
            if key == m:
                if min(hist[m]) < 0:
                    lns += ax1.plot(hist["epoch"], hist[m], "-", label="train " + m)
                else:
                    lns += ax1.semilogy(hist["epoch"], hist[m], "-", label="train " + m)
                color = lns[-1].get_color()
            elif key == "val_" + m:
                if min(hist[m]) < 0:
                    lns += ax1.plot(
                        hist["epoch"], hist["val_" + m], "--", c=color, label="val " + m
                    )
                else:
                    lns += ax1.semilogy(
                        hist["epoch"], hist["val_" + m], "--", c=color, label="val " + m
                    )
            elif key == "test_" + m:
                if min(hist[m]) < 0:
                    lns += ax1.plot(
                        hist["epoch"],
                        hist["test_" + m] * np.ones(numepo)
                        if isinstance(hist["test_" + m], (float, np.generic))
                        else hist["test_" + m],
                        ":",
                        c=color,
                        label="test " + m,
                    )
                else:
                    lns += ax1.semilogy(
                        hist["epoch"],
                        hist["test_" + m] * np.ones(numepo)
                        if isinstance(hist["test_" + m], (float, np.generic))
                        else hist["test_" + m],
                        ":",
                        c=color,
                        label="test " + m,
                    )
            if count == len(metrics1) - 1:
                for i in range(len(metrics2)):
                    if key == metrics2[i]:
                        lns1 += ax2.semilogy(
                            hist["epoch"], hist[key], ".", label=metrics2_label[i]
                        )
                if ylabel2:
                    ax2.set_ylabel(ylabel2)
        count += 1
    lns = lns + lns1
    labs = [l.get_label() for l in lns]
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(lns, labs, prop={"size": legend_fontsize})
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath, dpi=dpi)

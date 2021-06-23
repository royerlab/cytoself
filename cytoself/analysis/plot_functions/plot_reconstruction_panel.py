import os
import numpy as np
import matplotlib.pyplot as plt

from cytoself.analysis.plot_functions.batch2grid import convert_batch_to_image_grid


def plot_reconstruction_panel(
    train_data,
    test_data,
    gen_train,
    gen_test,
    savepath=None,
    filename_suffix="",
    grid_size=(4, 8),
    titles=None,
):
    if titles is None or len(titles) != 4:
        titles = [
            "training data originals",
            "test data originals",
            "training data reconstructions",
            "test data reconstructions",
        ]
    if filename_suffix != "":
        filename_suffix = "_" + filename_suffix
    num_img = min(
        train_data.shape[0], test_data.shape[0], gen_train.shape[0], gen_test.shape[0]
    )
    if grid_size[0] * grid_size[1] != num_img:
        raise ValueError("grid size does not match number of images.")
    # output reconstructed images
    f = plt.figure(figsize=(11, 6), dpi=300)  # (width, height)
    ax = f.add_subplot(2, 2, 1)
    ax.imshow(
        convert_batch_to_image_grid(train_data[:num_img], grid_size),
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_title(titles[0])
    plt.axis("off")

    ax = f.add_subplot(2, 2, 2)
    ax.imshow(
        convert_batch_to_image_grid(gen_train, grid_size),
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_title(titles[2])
    plt.axis("off")

    ax = f.add_subplot(2, 2, 3)
    ax.imshow(
        convert_batch_to_image_grid(test_data[:num_img], grid_size),
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_title(titles[1])
    plt.axis("off")

    ax = f.add_subplot(2, 2, 4)
    ax.imshow(
        convert_batch_to_image_grid(gen_test, grid_size),
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_title(titles[3])
    plt.axis("off")
    plt.subplots_adjust(
        left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1, wspace=0.045
    )
    if savepath:
        plt.savefig(
            os.path.join(savepath, "reconst_panel" + filename_suffix + ".png"), dpi=300
        )
        plt.close()
    else:
        plt.show()

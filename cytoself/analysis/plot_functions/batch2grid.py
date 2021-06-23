import numpy as np


def convert_batch_to_image_grid(
    image_batch, grid_size=(4, 8), zero_bottom=True,
):
    assert image_batch.shape[0] == np.prod(
        grid_size
    ), "Number of images and grid_size are inconsistent."
    image_batch = image_batch.squeeze()
    image_dim = image_batch.shape[1:]
    reshaped = (
        image_batch.reshape(grid_size + image_dim)
        .transpose(0, 2, 1, 3)
        .reshape(grid_size[0] * image_dim[0], grid_size[1] * image_dim[1])
    )
    if zero_bottom:
        reshaped = reshaped - np.amin(reshaped)
    return reshaped

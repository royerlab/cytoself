import os
import re
import mmap
import random
from tqdm import tqdm
from warnings import warn
from copy import deepcopy, copy
from joblib import Parallel, delayed

import numpy as np
import pandas as pd


def image_and_label_generator(
    image,
    label,
    n_label_out,
    batch_size=32,
    flip_dimension=False,
    rot90=False,
    slice2d=None,
    total_n_generate=None,
    no_repeat=False,
    **kwargs,
):
    """
    Image augmentation and label generation for model.fit
    :param image: input image; 2D or 3D
    :param label: input label
    :param n_label_out: number of label output
    :param batch_size: batch size
    :param flip_dimension: tuple of bool; has to consist with image spacial imput dimension
    :param rot90: random rotation 90 degrees on XY dimension but not on Z dimension.
    :param slice2d: returns 2D images by extracting a slice from a 3D image; int or 'random'
    :param total_n_generate: Total number of sample to generate.
    :param no_repeat: Do not yield the same data twice if True.
    :return: generator of augmented image
    """
    if isinstance(flip_dimension, bool):
        flip_dimension = [flip_dimension for _ in range(len(image.shape[1:-1]))]
    elif len(image.shape[1:-1]) != len(flip_dimension):
        raise ValueError(
            "flip_dimension has to be consistent with spatial dimension size of image."
        )
    label_only = False
    if "out_label_only" in kwargs:
        label_only = kwargs["out_label_only"]

    zdim = 0
    if slice2d is not None:
        if len(image.shape) == 5:
            zdim = image.shape[1]
            flip_dimension = flip_dimension[1:]
        else:
            warn("slice2d is ignored because the input image is not a 3D stack.")

        if isinstance(slice2d, int):
            if not 0 <= slice2d < zdim:
                raise ValueError("slice2d has to be 0 <= slice2d < image.shape[1].")
        elif slice2d != "random":
            raise ValueError(
                'slice2d has to be "random" or int such that 0 <= slice2d < batch_size.'
            )
    image = deepcopy(image)

    j = 0
    flow = True
    if total_n_generate is not None:
        image = image[:total_n_generate]
        label = label[:total_n_generate]
    num_cycle = np.ceil(image.shape[0] / batch_size)

    while flow:
        i = np.mod(j, num_cycle).astype(int)
        img_output = image[batch_size * i : batch_size * (i + 1)]
        lab_output = label[batch_size * i : batch_size * (i + 1)]

        if zdim != 0:
            if slice2d == "random":
                img_output = img_output[:, random.randint(0, zdim - 1), ...]
            else:
                img_output = img_output[:, slice2d, ...]
            # img_output = img_output[:, 0, ...]

        # Randomly flip each image
        for k in range(img_output.shape[0]):
            flip_index = [
                slice(None, None, random.choice([-1, None]))
                if i
                else slice(None, None, None)
                for i in flip_dimension
            ]
            flip_index = flip_index + [slice(None, None, None)]
            img = img_output[k]
            img = img[tuple(flip_index)]
            if rot90:
                img = np.rot90(
                    img,
                    random.randint(0, 3),
                    axes=(len(flip_dimension) - 2, len(flip_dimension) - 1),
                )
            img_output[k] = img

        j += 1
        flow = False if j > num_cycle and no_repeat else True
        if label_only:
            output_all = (img_output,) + (lab_output,) * n_label_out
        else:
            output_all = (img_output,) * 2 + (lab_output,) * n_label_out
        yield output_all[0], output_all[1:]

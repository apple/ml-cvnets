#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Optional
from utils.math_utils import make_divisible
import numpy as np


def _image_batch_pairs(crop_size_w: int,
                       crop_size_h: int,
                       batch_size_gpu0: int,
                       n_gpus: int,
                       max_scales: Optional[float] = 5,
                       check_scale_div_factor: Optional[int] = 32,
                       min_crop_size_w: Optional[int] = 160,
                       max_crop_size_w: Optional[int] = 320,
                       min_crop_size_h: Optional[int] = 160,
                       max_crop_size_h: Optional[int] = 320,
                       *args, **kwargs) -> list:
    """
        This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    :param crop_size_w: Base Image width (e.g., 224)
    :param crop_size_h: Base Image height (e.g., 224)
    :param batch_size_gpu0: Batch size on GPU 0 for base image
    :param n_gpus: Number of available GPUs
    :param max_scales: Number of scales. How many image sizes that we want to generate between min and max scale factors.
    :param check_scale_div_factor: Check if image scales are divisible by this factor.
    :param min_crop_size_w: Min. crop size along width
    :param max_crop_size_w: Max. crop size along width
    :param min_crop_size_h: Min. crop size along height
    :param max_crop_size_h: Max. crop size along height
    :param args:
    :param kwargs:
    :return: a sorted list of tuples. Each index is of the form (h, w, batch_size)
    """

    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)

    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)

    image_scales = set()

    for h, w in zip(height_dims, width_dims):
        # ensure that sampled sizes are divisible by check_scale_div_factor
        # This is important in some cases where input undergoes a fixed number of down-sampling stages
        # for instance, in ImageNet training, CNNs usually have 5 downsampling stages, which downsamples the
        # input image of resolution 224x224 to 7x7 size
        h = make_divisible(h, check_scale_div_factor)
        w = make_divisible(w, check_scale_div_factor)
        image_scales.add((h, w))

    image_scales = list(image_scales)

    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in image_scales:
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(batch_size_gpu0, int(round(n_elements/(crop_h * crop_y), 2)))

        _bsz = make_divisible(_bsz, n_gpus)
        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)

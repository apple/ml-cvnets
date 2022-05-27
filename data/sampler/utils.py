#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Optional, List
import numpy as np

from utils.math_utils import make_divisible


def _image_batch_pairs(
    crop_size_w: int,
    crop_size_h: int,
    batch_size_gpu0: int,
    n_gpus: int,
    max_scales: Optional[float] = 5,
    check_scale_div_factor: Optional[int] = 32,
    min_crop_size_w: Optional[int] = 160,
    max_crop_size_w: Optional[int] = 320,
    min_crop_size_h: Optional[int] = 160,
    max_crop_size_h: Optional[int] = 320,
    *args,
    **kwargs
) -> List:
    """
    This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    Args:
        crop_size_w (int): Base Image width (e.g., 224)
        crop_size_h (int): Base Image height (e.g., 224)
        batch_size_gpu0 (int): Batch size on GPU 0 for base image
        n_gpus (int): Number of available GPUs
        max_scales (Optional[int]): Number of scales. How many image sizes that we want to generate between min and max scale factors. Default: 5
        check_scale_div_factor (Optional[int]): Check if image scales are divisible by this factor. Default: 32
        min_crop_size_w (Optional[int]): Min. crop size along width. Default: 160
        max_crop_size_w (Optional[int]): Max. crop size along width. Default: 320
        min_crop_size_h (Optional[int]): Min. crop size along height. Default: 160
        max_crop_size_h (Optional[int]): Max. crop size along height. Default: 320

    Returns:
        a sorted list of tuples. Each index is of the form (h, w, batch_size)

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
        _bsz = max(1, int(round(n_elements / (crop_h * crop_y), 2)))

        _bsz = make_divisible(_bsz, n_gpus)
        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


def make_video_pairs(
    crop_size_h: int,
    crop_size_w: int,
    min_crop_size_h: int,
    max_crop_size_h: int,
    min_crop_size_w: int,
    max_crop_size_w: int,
    default_frames: int,
    max_scales: Optional[int] = 5,
    check_scale_div_factor: Optional[int] = 32,
    *args,
    **kwargs
) -> List:
    """
    This function creates number of frames and spatial size pairs for videos.

    Args:
        crop_size_h (int): Base Image height (e.g., 224)
        crop_size_w (int): Base Image width (e.g., 224)
        min_crop_size_w (int): Min. crop size along width.
        max_crop_size_w (int): Max. crop size along width.
        min_crop_size_h (int): Min. crop size along height.
        max_crop_size_h (int): Max. crop size along height.
        default_frames (int): Default number of frames per clip in a video.
        max_scales (Optional[int]): Number of scales. Default: 5
        check_scale_div_factor (Optional[int]): Check if spatial scales are divisible by this factor. Default: 32
    Returns:
        a sorted list of tuples. Each index is of the form (h, w, n_frames)
    """

    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)
    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)

    # ensure that spatial dimensions are divisible by check_scale_div_factor
    width_dims = [make_divisible(w, check_scale_div_factor) for w in width_dims]
    height_dims = [make_divisible(h, check_scale_div_factor) for h in height_dims]
    batch_pairs = set()
    n_elements = crop_size_w * crop_size_h * default_frames
    for (h, w) in zip(height_dims, width_dims):
        n_frames = max(1, int(round(n_elements / (h * w), 2)))
        batch_pairs.add((h, w, n_frames))
    return sorted(list(batch_pairs))

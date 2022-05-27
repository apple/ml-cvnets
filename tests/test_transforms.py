#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
import sys
import time
import random

sys.path.append("..")

import torch
import argparse
from PIL import Image
import numpy as np
import cv2

from data.transforms.image_pil import RandomCrop


def test_transform(opts):
    img_path = getattr(opts, "image_path", None)
    mask_path = getattr(opts, "mask_path", None)
    assert img_path is not None

    # setattr(opts, "image_augmentation.random_short_size_resize.short_side_min", 300)
    # setattr(opts, "image_augmentation.random_short_size_resize.short_side_max", 600)
    # setattr(opts, "image_augmentation.random_short_size_resize.interpolation", "bilinear")

    img = Image.open(img_path).convert("RGB")
    data = {"image": img}
    if mask_path is not None:
        mask = Image.open(mask_path)
        data["mask"] = mask

    for k in data.keys():
        cv2.imshow(f"{k}_t", np.array(data[k]))

    w, h = img.size

    transform = RandomCrop(opts, size=[512, 640])
    print(transform)

    for i in range(20):
        data_t = transform(copy.deepcopy(data))

        for k in data_t.keys():
            cv2.imshow(f"{k}_t", np.array(data_t[k]))
        cv2.waitKey()


if __name__ == "__main__":
    from options.opts import get_eval_arguments, load_config_file

    parser = get_eval_arguments(parse_args=False)

    parser.add_argument(
        "--image-path", type=str, default=None, help="Location of image path"
    )
    parser.add_argument(
        "--mask-path", type=str, default=None, help="Location of image path"
    )

    opts = parser.parse_args()
    opts = load_config_file(opts)
    test_transform(opts)

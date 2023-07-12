#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from cvnets.misc.common import freeze_modules_based_on_opts, get_tensor_sizes


def test_freeze_modules_based_on_opts() -> None:
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 20, 5)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(20, 64, 5)),
                ("relu2", nn.ReLU()),
            ]
        )
    )
    opts = argparse.Namespace(**{"model.freeze_modules": "conv1"})
    freeze_modules_based_on_opts(opts, model)

    model.train()
    assert model.conv1.training == False
    assert model.conv2.training == True
    assert model.relu1.training == True


def test_get_tensor_sizes() -> None:
    in_width = 224
    in_height = 224
    in_channels = 3
    in_batch_size = 1
    img = torch.randn(size=(in_batch_size, in_channels, in_height, in_width))

    # test for Tensor
    size_info = get_tensor_sizes(img)
    assert size_info == [in_batch_size, in_channels, in_height, in_width]

    # test for empty dict
    data_dict = {}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == []

    # test for dict with single key
    data_dict = {"image": img}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]")
    ]

    # test for dict with two keys
    data_dict = {"image_1": img, "image_2": img}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image_1: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
        str(f"image_2: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
    ]

    # test for nested dict
    data_dict = {"image_1": img, "image_2": {"image": img}}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image_1: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
        str(
            f"image_2: ['image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]']"
        ),
    ]

    # test for nested dict with non-tensor
    data_dict = {"image": img, "random_key": "data"}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]")
    ]

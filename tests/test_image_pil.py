#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pytest
import torch
from data.transforms import image_pil


def test_to_tensor() -> None:
    to_tensor = image_pil.ToTensor([])

    H, W, C = 2, 2, 3
    num_masks = 2
    data = {
        "image": torch.rand([H, W, C]),
        "mask": torch.randint(0, 1, [num_masks, H, W]),
    }

    output = to_tensor(data)

    assert output["image"].shape == (H, W, C)
    assert output["mask"].shape == (num_masks, H, W)


def test_to_tensor_bad_mask() -> None:
    to_tensor = image_pil.ToTensor([])

    H, W, C = 2, 2, 3
    num_categories = 2
    data = {
        "image": torch.rand([H, W, C]),
        "mask": torch.randint(0, 1, [num_categories, 1, H, W]),
    }

    with pytest.raises(SystemExit):
        to_tensor(data)

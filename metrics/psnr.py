#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
from typing import Optional

from . import register_stats_fn


@register_stats_fn(name="psnr")
def compute_psnr(
    prediction: Tensor, target: Tensor, no_uint8_conversion: Optional[bool] = False
) -> Tensor:

    if not no_uint8_conversion:
        prediction = prediction.mul(255.0).to(torch.uint8)
        target = target.mul(255.0).to(torch.uint8)
        MAX_I = 255 ** 2
    else:
        MAX_I = 1

    error = torch.pow(prediction - target, 2).float()
    mse = torch.mean(error) + 1e-10
    psnr = 10.0 * torch.log10(MAX_I / mse)
    return psnr

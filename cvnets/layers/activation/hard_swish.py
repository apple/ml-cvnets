#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, Optional
from . import register_act_fn


@register_act_fn(name="hard_swish")
class Hardswish(nn.Hardswish):
    """
    Applies the HardSwish function, as described in the paper
    `Searching for MobileNetv3 <https://arxiv.org/abs/1905.02244>`_
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(F, "hardswish"):
            return F.hardswish(input, self.inplace)
        else:
            x_hard_sig = F.relu(input + 3) / 6
            return input * x_hard_sig

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

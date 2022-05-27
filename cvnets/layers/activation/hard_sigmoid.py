#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple

from . import register_act_fn


@register_act_fn(name="hard_sigmoid")
class Hardsigmoid(nn.Hardsigmoid):
    """
    Applies the `Hard Sigmoid <https://arxiv.org/abs/1511.00363v3>`_ function
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(F, "hardsigmoid"):
            return F.hardsigmoid(input, self.inplace)
        else:
            return F.relu(input + 3) / 6

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

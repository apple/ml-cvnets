#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple

from . import register_act_fn


@register_act_fn(name="swish")
class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

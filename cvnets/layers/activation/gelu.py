#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Tuple

from . import register_act_fn


@register_act_fn(name="gelu")
class GELU(nn.GELU):
    """
    Applies the `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ function
    """

    def __init__(self) -> None:
        super().__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

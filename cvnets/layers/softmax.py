#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple


class Softmax(nn.Softmax):
    """
    Applies the Softmax function to an input tensor along the specified dimension

    Args:
        dim (int): Dimension along which softmax to be applied. Default: -1

    Shape:
        - Input: :math:`(*)` where :math:`*` is one or more dimensions
        - Output: same shape as the input
    """

    def __init__(self, dim: Optional[int] = -1, *args, **kwargs):
        super().__init__(dim=dim)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

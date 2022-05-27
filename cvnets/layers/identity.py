#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from typing import Tuple

from .base_layer import BaseLayer


class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        return x, 0.0, 0.0

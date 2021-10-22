#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor

from .base_layer import BaseLayer


class Identity(BaseLayer):
    def __init__(self):
        """
            Identity operator
        """
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> (Tensor, float, float):
        return x, 0.0, 0.0

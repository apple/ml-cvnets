#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import Tensor

from cvnets.layers.base_layer import BaseLayer


class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

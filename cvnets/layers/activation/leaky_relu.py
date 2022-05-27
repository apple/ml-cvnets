#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Tuple, Optional

from . import register_act_fn


@register_act_fn(name="leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    """
    Applies a leaky relu function. See `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    for more details.
    """

    def __init__(
        self, negative_slope: Optional[float] = 1e-2, inplace: Optional[bool] = False
    ) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

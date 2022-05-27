#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple

from . import register_act_fn


@register_act_fn(name="prelu")
class PReLU(nn.PReLU):
    """
    Applies the `Parametric Rectified Linear Unit <https://arxiv.org/abs/1502.01852>`_ function
    """

    def __init__(
        self, num_parameters: Optional[int] = 1, init: Optional[float] = 0.25
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

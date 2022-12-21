#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from typing import Tuple
from torchvision.ops import StochasticDepth as StochasticDepthTorch


class StochasticDepth(StochasticDepthTorch):
    """
    Implements the Stochastic Depth `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__(p=p, mode=mode)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0

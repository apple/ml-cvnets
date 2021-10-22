#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional

from . import register_norm_fn


@register_norm_fn(name="group_norm")
class GroupNorm(nn.GroupNorm):
    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 eps: Optional[float] = 1e-5,
                 affine: Optional[bool] = True
                 ):
        super(GroupNorm, self).__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

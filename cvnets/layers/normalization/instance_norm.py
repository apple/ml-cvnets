#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional

from . import register_norm_fn


@register_norm_fn(name="instance_norm")
@register_norm_fn(name="instance_norm_2d")
class InstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True
                 ):
        super(InstanceNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                             track_running_stats=track_running_stats)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


@register_norm_fn(name="instance_norm_1d")
class InstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True
                 ):
        super(InstanceNorm1d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                             track_running_stats=track_running_stats)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

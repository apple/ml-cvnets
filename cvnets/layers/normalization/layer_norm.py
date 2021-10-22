#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor, Size
from typing import Optional, Union, List

from . import register_norm_fn


@register_norm_fn(name="layer_norm")
class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape: Union[int, List[int], Size],
                 eps: Optional[float] = 1e-5,
                 elementwise_affine: Optional[bool] = True
                 ):
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

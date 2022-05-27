#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple

from . import register_norm_fn


@register_norm_fn(name="group_norm")
class GroupNorm(nn.GroupNorm):
    """
    Applies a `Group Normalization <https://arxiv.org/abs/1803.08494>`_ over an input tensor

    Args:
        num_groups (int): number of groups to separate the input channels into
        num_channels (int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        and :math:`*` is the remaining dimensions of the input tensor
        - Output: same shape as the input

    .. note::
        GroupNorm is the same as LayerNorm when `num_groups=1` and it is the same as InstanceNorm when
        `num_groups=C`.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: Optional[float] = 1e-5,
        affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

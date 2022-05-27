#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple

from . import register_norm_fn


@register_norm_fn(name="sync_batch_norm")
class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies a `Syncronized Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over the input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`*` is the remaining input dimensions
        - Output: same shape as the input

    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
import argparse

from . import register_classification_loss_fn
from .. import BaseCriteria


@register_classification_loss_fn(name="binary_cross_entropy")
class ClsBinaryCrossEntropy(BaseCriteria):
    """Binary CE for classification tasks"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

    def forward(
        self, input_sample: Tensor, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        if target.dim() != prediction.dim():
            target = F.one_hot(target, num_classes=prediction.shape[-1])

        return F.binary_cross_entropy_with_logits(
            input=prediction,
            target=target.to(prediction.dtype),
            weight=None,
            reduction="sum",
        )

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)

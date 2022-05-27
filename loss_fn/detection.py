#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from typing import Union

from . import BaseCriteria, register_loss_fn
from .detection_loss_fns import get_detection_loss, arguments_detection_loss_fn


@register_loss_fn("detection")
class DetectionLoss(BaseCriteria):
    def __init__(self, opts):
        super(DetectionLoss, self).__init__()

        self.criteria = get_detection_loss(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.detection.name",
            type=str,
            default="cross_entropy",
            help="Detection loss function name",
        )

        parser = arguments_detection_loss_fn(parser)
        return parser

    def forward(
        self,
        input_sample: Tensor,
        prediction: Union[Tensor, Union[Tensor, Tensor]],
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:

        loss = self.criteria(
            input_sample=input_sample, prediction=prediction, target=target
        )
        return loss

    def __repr__(self):
        return self.criteria.__repr__()

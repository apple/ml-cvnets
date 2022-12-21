#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger
from typing import Any

from . import BaseCriteria, register_loss_fn
from .segmentation_loss_fns import get_segmentation_loss, arguments_seg_loss_fn


@register_loss_fn("segmentation")
class SegmentationLoss(BaseCriteria):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.criteria = get_segmentation_loss(opts=opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.segmentation.name",
            type=str,
            default="cross_entropy",
            help="Segmentation loss function name",
        )
        parser = arguments_seg_loss_fn(parser=parser)
        return parser

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target,
            *args,
            **kwargs
        )

    def __repr__(self):
        return self.criteria.__repr__()

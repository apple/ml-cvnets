#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger

from . import BaseCriteria, register_loss_fn
from .classification_loss_fns import get_classification_loss, arguments_cls_loss_fn


@register_loss_fn("classification")
class ClassificationLoss(BaseCriteria):
    def __init__(self, opts):
        super(ClassificationLoss, self).__init__()

        self.criteria = get_classification_loss(opts=opts)

    def forward(
        self, input_sample: Tensor, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        return self.criteria(
            input_sample=input_sample, prediction=prediction, target=target
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.classification.name",
            type=str,
            default="cross_entropy",
            help="Loss function name",
        )
        parser = arguments_cls_loss_fn(parser)
        return parser

    def __repr__(self):
        return self.criteria.__repr__()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger

from . import BaseCriteria, register_loss_fn
from .classification_loss_fns import ClsCrossEntropy, LabelSmoothing, SUPPORTED_CLS_LOSS_FNS


@register_loss_fn("classification")
class ClassificationLoss(BaseCriteria):
    def __init__(self, opts):
        loss_fn_name = getattr(opts, "loss.classification.name", "cross_entropy")
        super(ClassificationLoss, self).__init__()

        if loss_fn_name == "cross_entropy":
            self.criteria = ClsCrossEntropy(opts=opts)
        elif loss_fn_name == "label_smoothing":
            self.criteria = LabelSmoothing(opts=opts)
        else:
            temp_str = "Loss function ({}) not yet supported. " \
                       "\n Supported classification loss functions are:".format(loss_fn_name)
            for i, m_name in enumerate(SUPPORTED_CLS_LOSS_FNS):
                temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
            logger.error(temp_str)

    def forward(self, input_sample: Tensor, prediction: Tensor, target: Tensor) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.classification.name", type=str, default="cross_entropy", help="Loss function name")
        parser = ClsCrossEntropy.add_arguments(parser=parser)
        parser = LabelSmoothing.add_arguments(parser=parser)
        return parser

    def __repr__(self):
        return self.criteria.__repr__()


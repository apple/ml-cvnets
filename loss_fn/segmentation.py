#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger
from typing import Union, Tuple

from . import BaseCriteria, register_loss_fn
from .segmentation_loss_fns import SegCrossEntropy, SUPPORTED_SEG_LOSS_FNS


@register_loss_fn("segmentation")
class SegmentationLoss(BaseCriteria):
    def __init__(self, opts):
        loss_fn_name = getattr(opts, "loss.segmentation.name", "cross_entropy")
        super(SegmentationLoss, self).__init__()
        if loss_fn_name == "cross_entropy":
            self.criteria = SegCrossEntropy(opts=opts)
        else:
            temp_str = "Loss function ({}) not yet supported. " \
                       "\n Supported segmentation loss functions are:".format(loss_fn_name)
            for i, m_name in enumerate(SUPPORTED_SEG_LOSS_FNS):
                temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
            logger.error(temp_str)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.segmentation.name", type=str, default="cross_entropy",
                           help="Segmentation loss function name")
        parser = SegCrossEntropy.add_arguments(parser=parser)
        return parser

    def forward(self, input_sample: Tensor, prediction: Union[Tensor, Tuple[Tensor, Tensor]], target: Tensor) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target
        )

    def __repr__(self):
        return self.criteria.__repr__()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger
from typing import Union

from . import BaseCriteria, register_loss_fn
from .detection_loss_fns import SSDLoss, SUPPORTED_DETECTION_LOSS_FNS


@register_loss_fn("detection")
class DetectionLoss(BaseCriteria):
    def __init__(self, opts):
        loss_fn_name = getattr(opts, "loss.detection.name", "cross_entropy")
        super(DetectionLoss, self).__init__()
        if loss_fn_name == "ssd_multibox_loss":
            self.criteria = SSDLoss(opts=opts)
        else:
            temp_str = "Loss function ({}) not yet supported. " \
                       "\n Supported detection loss functions are:".format(loss_fn_name)
            for i, m_name in enumerate(SUPPORTED_DETECTION_LOSS_FNS):
                temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
            logger.error(temp_str)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.detection.name", type=str, default="cross_entropy",
                           help="Segmentation loss function name")
        parser = SSDLoss.add_arguments(parser=parser)
        return parser

    def forward(self,
                input_sample: Tensor,
                prediction: Union[Tensor, Union[Tensor, Tensor]],
                target: Tensor) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target
        )

    def __repr__(self):
        return self.criteria.__repr__()

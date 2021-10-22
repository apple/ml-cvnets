#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger

from . import BaseCriteria, register_loss_fn
from .distillation_loss_fns import VanillaDistillationLoss, SUPPORTED_DISTILL_LOSS_FNS


@register_loss_fn("distillation")
class DistillationLoss(BaseCriteria):
    def __init__(self, opts):
        loss_fn_name = getattr(opts, "loss.distillation.name", "vanilla")
        super(DistillationLoss, self).__init__()
        if loss_fn_name == "vanilla":
            self.criteria = VanillaDistillationLoss(opts=opts)
        else:
            temp_str = "Loss function ({}) not yet supported. " \
                       "\n Supported distillation loss functions are:".format(loss_fn_name)
            for i, m_name in enumerate(SUPPORTED_DISTILL_LOSS_FNS):
                temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
            logger.error(temp_str)

    def forward(self,  input_sample: Tensor, prediction: Tensor, target: Tensor) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.distillation.name", type=str, default="vanilla",
                           help="Distillation loss function name")
        parser = VanillaDistillationLoss.add_arguments(parser=parser)
        return parser

    def __repr__(self):
        return self.criteria.__repr__()

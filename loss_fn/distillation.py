#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger

from . import BaseCriteria, register_loss_fn
from .distillation_loss_fns import get_distillation_loss, arguments_distill_loss_fn


@register_loss_fn("distillation")
class DistillationLoss(BaseCriteria):
    def __init__(self, opts, *args, **kwargs):
        loss_fn_name = getattr(opts, "loss.distillation.name", "vanilla")
        super().__init__(opts, *args, **kwargs)
        self.criteria = get_distillation_loss(opts=opts, *args, **kwargs)

    def forward(
        self, input_sample: Tensor, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        return self.criteria(
            input_sample=input_sample,
            prediction=prediction,
            target=target,
            *args,
            **kwargs
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.distillation.name",
            type=str,
            default="vanilla",
            help="Distillation loss function name",
        )
        parser = arguments_distill_loss_fn(parser=parser)
        return parser

    def extra_repr(self) -> str:
        if hasattr(self.criteria, "extra_repr"):
            return self.criteria.extra_repr()
        return ""

    def __repr__(self):
        return "{}({}\n)".format(self.criteria.__class__.__name__, self.extra_repr())

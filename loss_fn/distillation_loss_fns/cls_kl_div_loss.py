#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os.path
from torch.nn import functional as F
import torch
from torch import nn, Tensor
import argparse
from typing import Dict, Union

from . import register_distillation_loss_fn
from .. import BaseCriteria

from .utils import build_cls_teacher_from_opts


@register_distillation_loss_fn(name="cls_kl_div_loss")
class ClsKLDivLoss(BaseCriteria):
    """
    KL Loss for classification
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.teacher = build_cls_teacher_from_opts(opts=opts)
        self.temperature = getattr(
            opts, "loss.distillation.cls_kl_div_loss.temperature", 1.0
        )
        self.distillation_mode = getattr(
            opts, "loss.distillation.cls_kl_div_loss.mode", "soft"
        )
        self.topk = getattr(opts, "loss.distillation.cls_kl_div_loss.topk", 1)
        self.label_smoothing = getattr(
            opts, "loss.distillation.cls_kl_div_loss.label-smoothing", 0.0
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.distillation.cls-kl-div-loss.temperature",
            type=float,
            default=1.0,
            help="Temperature for KL Div. loss",
        )
        group.add_argument(
            "--loss.distillation.cls-kl-div-loss.mode",
            type=str,
            default="soft",
            help="Distillation mode",
        )
        group.add_argument(
            "--loss.distillation.cls-kl-div-loss.topk",
            type=int,
            default=1,
            help="Distill top-k labels from teacher when using hard-labels",
        )
        group.add_argument(
            "--loss.distillation.cls-kl-div-loss.label-smoothing",
            type=float,
            default=0.0,
            help="Use label smoothing when using hard-labels",
        )
        return parser

    def extra_repr(self) -> str:
        extra_repr_str = (
            f"\n\ttemperature={self.temperature}" f"\n\tmode={self.distillation_mode}"
        )
        if self.distillation_mode.find("hard") > -1:
            extra_repr_str += (
                f"\n\ttopk={self.topk}" f"\n\tlabel_smoothing={self.label_smoothing}"
            )
        return extra_repr_str

    def _forward_soft_labels(
        self, prediction: Tensor, teacher_logits: Tensor
    ) -> Tensor:
        with torch.no_grad():
            teacher_lprobs = F.log_softmax(
                teacher_logits / self.temperature, dim=1
            ).detach()

        student_lprobs = F.log_softmax(prediction / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_lprobs, teacher_lprobs, reduction="batchmean", log_target=True
        )
        return kl_loss * (self.temperature**2)

    def _forward_hard_labels(
        self, prediction: Tensor, teacher_logits: Tensor
    ) -> Tensor:
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
            _, teacher_topk_labels = torch.topk(
                teacher_probs, k=self.topk, dim=-1, largest=True, sorted=True
            )

        if self.topk > 1:
            num_classes = prediction.shape[-1]
            teacher_topk_labels = F.one_hot(
                teacher_topk_labels, num_classes=num_classes
            )
            teacher_topk_labels = teacher_topk_labels.sum(1)
            teacher_topk_labels = teacher_topk_labels.to(dtype=prediction.dtype)

            # smooth labels corresponding to multiple classes
            smooth_class_p = (1.0 - self.label_smoothing) / self.topk
            # distribute the mass over remaining classes
            smooth_non_class_p = self.label_smoothing / (num_classes - self.topk)

            teacher_topk_labels = torch.where(
                teacher_topk_labels == 1.0, smooth_class_p, smooth_non_class_p
            )

            # scale by number of classes. Otherwise, the contribution is small
            loss = (
                F.binary_cross_entropy_with_logits(
                    input=prediction, target=teacher_topk_labels, reduction="mean"
                )
                * num_classes
            )
        else:
            teacher_topk_labels = teacher_topk_labels.reshape(-1)
            loss = F.cross_entropy(
                input=prediction,
                target=teacher_topk_labels,
                reduction="mean",
                label_smoothing=self.label_smoothing,
            )
        return loss

    def forward(
        self, input_sample: Tensor, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:

        with torch.no_grad():
            self.teacher.eval()
            teacher_logits: Union[Tensor, Dict] = self.teacher(input_sample)
            # Dict in case of neural aug
            if isinstance(teacher_logits, Dict):
                teacher_logits = teacher_logits["logits"]

        if self.distillation_mode == "soft":
            return self._forward_soft_labels(
                prediction=prediction, teacher_logits=teacher_logits
            )
        elif self.distillation_mode == "hard":
            return self._forward_hard_labels(
                prediction=prediction, teacher_logits=teacher_logits
            )
        else:
            raise NotImplementedError

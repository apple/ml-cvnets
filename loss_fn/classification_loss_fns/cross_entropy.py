#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
import argparse

from . import register_classification_loss_fn
from .. import BaseCriteria


@register_classification_loss_fn(name="cross_entropy")
class ClsCrossEntropy(BaseCriteria):
    """Cross entropy for classification tasks"""

    def __init__(self, opts):
        ignore_idx = getattr(opts, "loss.ignore_idx", -1)
        use_class_wts = getattr(
            opts, "loss.classification.cross_entropy.class_weights", False
        )
        super(ClsCrossEntropy, self).__init__()

        self.ignore_idx = ignore_idx
        self.use_class_wts = use_class_wts
        self.label_smoothing = getattr(opts, "loss.classification.label_smoothing", 0.0)

    def forward(
        self, input_sample: Tensor, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        weight = None
        if self.use_class_wts and self.training:
            n_classes = prediction.shape[1]
            weight = self._class_weights(target=target, n_classes=n_classes)

        return F.cross_entropy(
            input=prediction,
            target=target,
            weight=weight,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing
            if self.training
            else 0.0,  # for validation, compute standard CE loss
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.classification.cross-entropy.class-weights",
            action="store_true",
            help="Use class weights in loss function",
        )
        group.add_argument(
            "--loss.classification.label-smoothing",
            type=float,
            default=0.0,
            help="Label smoothing value",
        )
        return parser

    def __repr__(self):
        return "{}(\n\tignore_idx={}\n\tclass_wts={}\n\tlabel_smoothing={}\n)".format(
            self.__class__.__name__,
            self.ignore_idx,
            self.use_class_wts,
            self.label_smoothing,
        )

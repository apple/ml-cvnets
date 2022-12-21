#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch.nn import functional as F
from torch import Tensor
import argparse
from typing import Tuple, Dict, List


from . import register_detection_loss_fn
from .. import BaseCriteria


@register_detection_loss_fn(name="mask_rcnn_loss")
class MaskRCNNLoss(BaseCriteria):
    """Mask RCNN Loss"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.classifier-weight",
            type=float,
            default=1,
            help="Weight for classifier.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.box-reg-weight",
            type=float,
            default=1,
            help="Weight for box reg.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.mask-weight",
            type=float,
            default=1,
            help="Weight for mask.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.objectness-weight",
            type=float,
            default=1,
            help="Weight for objectness.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.rpn-box-reg",
            type=float,
            default=1,
            help="Weight for rpn box reg.",
        )
        return parser

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.classifier_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.classifier_weight"
        )
        self.box_reg_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.box_reg_weight"
        )
        self.mask_weight = getattr(opts, "loss.detection.mask_rcnn_loss.mask_weight")
        self.objectness_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.objectness_weight"
        )
        self.rpn_box_reg = getattr(opts, "loss.detection.mask_rcnn_loss.rpn_box_reg")

    def extra_repr(self) -> str:
        return (
            f"\n\tclassifier_wt={self.classifier_weight}"
            f"\n\tbox_reg_weight={self.box_reg_weight}"
            f"\n\tmask_weight={self.mask_weight}"
            f"\n\tobjectness_weight={self.objectness_weight}"
            f"\n\trpn_box_reg={self.rpn_box_reg}"
        )

    def forward(
        self,
        input_sample: Dict[str, List],
        prediction: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:

        try:
            # Loss is computed inside the Mask RCNN model. Here, we only compute the weighted sum of
            # different loss functions.
            total_loss = (
                self.classifier_weight * prediction["loss_classifier"]
                + self.box_reg_weight * prediction["loss_box_reg"]
                + self.mask_weight * prediction["loss_mask"]
                + self.objectness_weight * prediction["loss_objectness"]
                + self.rpn_box_reg * prediction["loss_rpn_box_reg"]
            )
            return {"total_loss": total_loss, **prediction}
        except KeyError:
            # MaskRCNN doesn't return the loss during validation.
            device = input_sample["image"][0].device
            return {"total_loss": torch.tensor(0.0, device=device)}

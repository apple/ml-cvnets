#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch.nn import functional as F
from torch import Tensor
import argparse
from typing import Tuple, Dict

from utils.tensor_utils import tensor_to_python_float
from utils import logger
from utils.ddp_utils import is_master
from cvnets.misc.third_party.ssd_utils import hard_negative_mining

from . import register_detection_loss_fn
from .utils import sigmoid_focal_loss
from .. import BaseCriteria

from torchvision.models.detection.faster_rcnn import FasterRCNN


@register_detection_loss_fn(name="ssd_multibox_loss")
class SSDLoss(BaseCriteria):
    """SSD Loss"""

    def __init__(self, opts):
        super(SSDLoss, self).__init__()
        self.unscaled_reg_loss = 1e-7
        self.unscaled_conf_loss = 1e-7
        self.neg_pos_ratio = getattr(
            opts, "loss.detection.ssd_multibox_loss.neg_pos_ratio", 3
        )
        self.wt_loc = 1.0
        self.curr_iter = 0
        self.max_iter = getattr(
            opts, "loss.detection.ssd_multibox_loss.max_monitor_iter", -1
        )
        self.update_inter = getattr(
            opts, "loss.detection.ssd_multibox_loss.update_wt_freq", 200
        )
        self.is_distributed = getattr(opts, "ddp.use_distributed", False)
        self.is_master = is_master(opts)
        self.label_smoothing = getattr(
            opts, "loss.detection.ssd_multibox_loss.label_smoothing", 0.0
        )
        if not (0.0 <= self.label_smoothing < 1.0):
            logger.error(
                "The value of --loss.detection.ssd-multibox-loss.label-smoothing should be between 0 and 1. "
                "Got: {}".format(self.label_smoothing)
            )

        self.reset_unscaled_loss_values()

    def reset_unscaled_loss_values(self):
        # initialize with very small float values
        self.unscaled_conf_loss = 1e-7
        self.unscaled_reg_loss = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.neg-pos-ratio",
            type=int,
            default=3,
            help="Negative positive ratio in SSD loss",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.max-monitor-iter",
            type=int,
            default=-1,
            help="Number of iterations for monitoring location and classification loss.",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.update-wt-freq",
            type=int,
            default=200,
            help="Update the weights after N number of iterations",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.label-smoothing",
            type=float,
            default=0.0,
            help="Label smoothing for classification labels in SSD",
        )
        return parser

    def __repr__(self):
        return "{}(\n\tneg_pos_ratio={}\n\tbox_loss=SmoothL1\n\tclass_loss=CrossEntropy\n\twt_loss={}\n)".format(
            self.__class__.__name__,
            self.neg_pos_ratio,
            True if self.max_iter > 0 else False,
        )

    def _forward_detection_loss(
        self, prediction: Dict, target: Dict, *args, **kwargs
    ) -> Tensor:
        # confidence: (batch_size, num_priors, num_classes)
        # predicted_locations :(batch_size, num_priors, 4)

        confidence = prediction["scores"]
        predicted_locations = prediction["boxes"]

        gt_labels = target["box_labels"]
        gt_locations = target["box_coordinates"]

        num_classes = confidence.shape[-1]
        num_coordinates = predicted_locations.shape[-1]

        pos_mask = gt_labels > 0
        predicted_locations = predicted_locations[pos_mask].reshape(-1, num_coordinates)
        gt_locations = gt_locations[pos_mask].reshape(-1, num_coordinates)
        num_pos = max(1, gt_locations.shape[0])
        smooth_l1_loss = F.smooth_l1_loss(
            predicted_locations, gt_locations, reduction="sum"
        )

        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        label_smoothing = self.label_smoothing if self.training else 0.0
        classification_loss = F.cross_entropy(
            input=confidence.reshape(-1, num_classes),
            target=gt_labels[mask],
            reduction="sum",
            label_smoothing=label_smoothing,
        )

        if self.curr_iter <= self.max_iter and self.training:
            # classification loss may dominate localization loss or vice-versa
            # therefore, to ensure that their contributions are equal towards total loss, we scale regression loss.
            # if classification loss contribution is less (or more), then scaling factor will be < 1 ( > 1)
            self.unscaled_conf_loss += tensor_to_python_float(
                classification_loss, is_distributed=self.is_distributed
            )
            self.unscaled_reg_loss += tensor_to_python_float(
                smooth_l1_loss, is_distributed=self.is_distributed
            )

            if (
                self.curr_iter + 1
            ) % self.update_inter == 0 or self.curr_iter == self.max_iter:
                before_update = round(tensor_to_python_float(self.wt_loc), 4)
                self.wt_loc = self.unscaled_conf_loss / self.unscaled_reg_loss
                self.reset_unscaled_loss_values()

                if self.is_master:
                    after_update = round(tensor_to_python_float(self.wt_loc), 4)
                    logger.log(
                        f"Updating localization loss multiplier from {before_update} to {after_update}"
                    )

            self.curr_iter += 1

        if self.training and self.wt_loc > 0.0:
            smooth_l1_loss = smooth_l1_loss * self.wt_loc

        return (smooth_l1_loss + classification_loss) / num_pos

    def forward(
        self, input_sample: Tensor, prediction: Dict, target: Dict, *args, **kwargs
    ) -> Tensor:
        # confidence: (batch_size, num_priors, num_classes)
        # predicted_locations :(batch_size, num_priors, 4)

        detection_loss = self._forward_detection_loss(
            prediction=prediction, target=target
        )
        return detection_loss

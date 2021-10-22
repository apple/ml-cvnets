#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
import numpy as np
from typing import Optional, Union, Tuple

from .third_party.ssd_utils import assign_priors
from cvnets.misc.box_utils import (
    center_form_to_corner_form,
    corner_form_to_center_form,
    convert_boxes_to_locations
)


class SSDMatcher(object):
    '''
        Match priors with ground truth boxes
    '''
    def __init__(self,
                 center_variance: Optional[float] = 0.1,
                 size_variance: Optional[float] = 0.2,
                 iou_threshold: Optional[float] = 0.5) -> None:
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self,
                 gt_boxes_cor: Union[np.ndarray, Tensor],
                 gt_labels: Union[np.ndarray, Tensor],
                 reference_boxes_ctr: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param gt_boxes_cor: Ground truth boxes in corner form (x1, y1, x2, y2)
        :param gt_labels: Ground truth box labels
        :param reference_boxes_ctr: Anchor boxes in center form (c_x1, c_y1, dw, dh)
        :return: Matched boxes and their corresponding labels in center form
        """

        if isinstance(gt_boxes_cor, np.ndarray):
            gt_boxes_cor = torch.from_numpy(gt_boxes_cor)
        if isinstance(gt_labels, np.ndarray):
            gt_labels = torch.from_numpy(gt_labels)

        # convert box priors from center [c_x, c_y] to corner_form [x, y]
        reference_boxes_cor = center_form_to_corner_form(boxes=reference_boxes_ctr)

        matched_boxes_cor, matched_labels = assign_priors(
            gt_boxes_cor, # gt_boxes are in corner form [x, y, w, h]
            gt_labels,
            reference_boxes_cor, # priors are in corner form [x, y, w, h]
            self.iou_threshold
        )

        # convert the matched boxes to center form [c_x, c_y]
        matched_boxes_ctr = corner_form_to_center_form(matched_boxes_cor)

        # Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf
        boxes_for_regression = convert_boxes_to_locations(
            gt_boxes=matched_boxes_ctr, # center form
            prior_boxes=reference_boxes_ctr, # center form
            center_variance=self.center_variance,
            size_variance=self.size_variance
        )

        return boxes_for_regression, matched_labels



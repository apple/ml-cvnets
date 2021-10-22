#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
from typing import Optional, Tuple, Union

from . import register_stats_fn


@register_stats_fn(name="iou")
def compute_miou_batch(prediction: Union[Tuple[Tensor, Tensor], Tensor], target: Tensor, epsilon: Optional[float] = 1e-7):
    if isinstance(prediction, Tuple) and len(prediction) == 2:
        mask = prediction[0]
        assert isinstance(mask, Tensor)
    elif isinstance(prediction, Tensor):
        mask = prediction
        assert isinstance(mask, Tensor)
    else:
        raise NotImplementedError(
            "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor")

    num_classes = mask.shape[1]
    pred_mask = torch.max(mask, dim=1)[1]
    assert pred_mask.dim() == 3, "Predicted mask tensor should be 3-dimensional (B x H x W)"

    pred_mask = pred_mask.byte()
    target = target.byte()

    # shift by 1 so that 255 is 0
    pred_mask += 1
    target += 1

    pred_mask = pred_mask * (target > 0)
    inter = pred_mask * (pred_mask == target)
    area_inter = torch.histc(inter.float(), bins=num_classes, min=1, max=num_classes)
    area_pred = torch.histc(pred_mask.float(), bins=num_classes, min=1, max=num_classes)
    area_mask = torch.histc(target.float(), bins=num_classes, min=1, max=num_classes)
    area_union = area_pred + area_mask - area_inter + epsilon
    return area_inter, area_union

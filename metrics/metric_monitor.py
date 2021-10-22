#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple
from torch import Tensor

from utils.tensor_utils import tensor_to_python_float

from .topk_accuracy import top_k_accuracy
from .intersection_over_union import compute_miou_batch


def metric_monitor(pred_label: Tensor or Tuple[Tensor], target_label: Tensor, loss: Tensor or float, metric_names: list,
                   use_distributed: Optional[bool] = False):
    metric_vals = dict()
    if "loss" in metric_names:
        loss = tensor_to_python_float(loss, is_distributed=use_distributed)
        metric_vals['loss'] = loss

    if "top1" in metric_names:
        top_1_acc, top_5_acc = top_k_accuracy(pred_label, target_label, top_k=(1, 5))
        top_1_acc = tensor_to_python_float(top_1_acc, is_distributed=use_distributed)
        metric_vals['top1'] = top_1_acc
        if "top5" in metric_names:
            top_5_acc = tensor_to_python_float(top_5_acc, is_distributed=use_distributed)
            metric_vals['top5'] = top_5_acc

    if "iou" in metric_names:
        inter, union = compute_miou_batch(prediction=pred_label, target=target_label)

        inter = tensor_to_python_float(inter, is_distributed=use_distributed)
        union = tensor_to_python_float(union, is_distributed=use_distributed)
        metric_vals['iou'] = {
            'inter': inter,
            'union': union
        }

    return metric_vals

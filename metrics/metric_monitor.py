#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from typing import Optional, Tuple, Any, Dict, Union
from numbers import Number

from utils.tensor_utils import tensor_to_python_float
from utils import logger

from .topk_accuracy import top_k_accuracy
from .intersection_over_union import compute_miou_batch
from .psnr import compute_psnr


def metric_monitor(
    opts,
    pred_label: Any,
    target_label: Any,
    loss: Tensor or float,
    metric_names: list,
    use_distributed: Optional[bool] = False,
    grad_norm: Optional = None,
    is_evaluation: Optional[bool] = False,
    *args,
    **kwargs
):
    """
    This function aggregate different metrics and convert them into floats, so that
    they can be easily consumed by stats.py file
    """
    metric_vals = dict()
    if "loss" in metric_names:
        metric_vals["loss"] = gather_loss(loss, is_distributed=use_distributed)

    if "grad_norm" in metric_names:
        metric_vals["grad_norm"] = gather_grad_norm(
            grad_norm, is_distributed=use_distributed
        )

    if "top1" in metric_names:
        top_1, top_5 = gather_top_k_metrics(
            prediction=pred_label, target=target_label, is_distributed=use_distributed
        )
        metric_vals["top1"] = top_1
        if "top5" in metric_names:
            metric_vals["top5"] = top_5

    if "iou" in metric_names:
        inter, union = gather_iou_metrics(
            prediction=pred_label, target=target_label, is_distributed=use_distributed
        )
        metric_vals["iou"] = {"inter": inter, "union": union}

    if "psnr" in metric_names:
        psnr = compute_psnr(prediction=pred_label, target=target_label)
        metric_vals["psnr"] = tensor_to_python_float(
            psnr, is_distributed=use_distributed
        )

    return metric_vals


def gather_loss(
    loss: Union[Tensor, Dict], is_distributed: bool
) -> Union[Number, Dict[str, Number]]:
    """
    This function gather losses from different processes and converts to float.
    """
    if isinstance(loss, (int, float)):
        return loss * 1.0
    elif isinstance(loss, Tensor):
        return tensor_to_python_float(loss, is_distributed=is_distributed)
    elif isinstance(loss, Dict):
        loss_dict = {}

        if "total_loss" not in list(loss.keys()):
            logger.error(
                "total_loss key is required for loss functions that return outputs as dictionary."
            )

        for k, v in loss.items():
            if v is None:
                continue
            v_float = tensor_to_python_float(v, is_distributed=is_distributed)
            loss_dict[k] = v_float
        return loss_dict
    else:
        logger.error("Metric monitor supports Tensor or Dict of Tensors")


def gather_grad_norm(
    grad_norm: Union[Tensor, Dict], is_distributed: bool
) -> Union[Number, Dict[str, Number]]:
    """
    This function gather grad_norm from different processes and converts to float.
    """
    if grad_norm is None:
        return 1e-7

    if isinstance(grad_norm, (int, float)):
        return grad_norm * 1.0
    if isinstance(grad_norm, Tensor):
        return tensor_to_python_float(grad_norm, is_distributed=is_distributed)
    elif isinstance(grad_norm, Dict):
        grad_norm_dict = {}
        for k, v in grad_norm.items():
            if v is None:
                continue
            v_float = tensor_to_python_float(v, is_distributed=is_distributed)
            grad_norm_dict[k] = v_float
        return grad_norm_dict
    else:
        logger.error("Metric monitor supports Tensor or Dict of Tensors")


def gather_top_k_metrics(
    prediction: Union[Tensor, Dict], target: Union[Tensor, Dict], is_distributed: bool
) -> Union[Tuple[Number, Number], Tuple[Dict[str, Number], Dict[str, Number]]]:
    """
    This function gather top-1 and top-5 metrics from different processes and converts to float.
    """
    # We have four combinations between prediction and target types:
    # 1. (Tensor, Tensor)
    # 2. (Dict, Tensor)
    # 3. (Dict, Dict)
    # 4. (Tensor, Dict) --> This combination is rare

    if isinstance(prediction, Tensor) and isinstance(target, Tensor):
        top_1_acc, top_5_acc = top_k_accuracy(prediction, target, top_k=(1, 5))
        top_1_acc = tensor_to_python_float(top_1_acc, is_distributed=is_distributed)
        top_5_acc = tensor_to_python_float(top_5_acc, is_distributed=is_distributed)
        return top_1_acc, top_5_acc
    elif isinstance(prediction, Dict) and isinstance(target, Tensor):
        top1_dict = {}
        top5_dict = {}
        for pred_k, pred_v in prediction.items():
            if (
                isinstance(pred_v, Tensor) and pred_v.dim() == 2 and target.dim() == 1
            ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                top_1_acc, top_5_acc = top_k_accuracy(pred_v, target, top_k=(1, 5))
                top_1_acc = tensor_to_python_float(
                    top_1_acc, is_distributed=is_distributed
                )
                top_5_acc = tensor_to_python_float(
                    top_5_acc, is_distributed=is_distributed
                )
                top1_dict[pred_k] = top_1_acc
                top5_dict[pred_k] = top_5_acc
        return top1_dict, top5_dict
    elif isinstance(prediction, Dict) and isinstance(target, Dict):
        # prediction and target dictionaries should have intersecting keys
        prediction_keys = prediction.keys()
        target_keys = target.keys()

        intersection_keys = list(set(prediction_keys).intersection(target_keys))
        if len(intersection_keys) == 0:
            logger.error(
                "The keys in prediction and target are different. "
                " Got: Prediction keys={} and Target keys={}".format(
                    prediction_keys, target_keys
                )
            )

        top1_dict = {}
        top5_dict = {}
        for pred_k in intersection_keys:
            pred_v = prediction[pred_k]
            target_v = target[pred_k]
            if (
                isinstance(pred_v, Tensor)
                and isinstance(target_v, Tensor)
                and pred_v.dim() == 2
                and target_v.dim() == 1
            ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                top_1_acc, top_5_acc = top_k_accuracy(pred_v, target_v, top_k=(1, 5))
                top_1_acc = tensor_to_python_float(
                    top_1_acc, is_distributed=is_distributed
                )
                top_5_acc = tensor_to_python_float(
                    top_5_acc, is_distributed=is_distributed
                )
                top1_dict[pred_k] = top_1_acc
                top5_dict[pred_k] = top_5_acc
        return top1_dict, top5_dict
    elif isinstance(prediction, Tensor) and isinstance(target, Dict):
        # rare but possible
        top1_dict = {}
        top5_dict = {}
        for target_k, target_v in target.items():
            if (
                isinstance(target_v, Tensor)
                and prediction.dim() == 2
                and target_v.dim() == 1
            ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                top_1_acc, top_5_acc = top_k_accuracy(
                    prediction, target_v, top_k=(1, 5)
                )
                top_1_acc = tensor_to_python_float(
                    top_1_acc, is_distributed=is_distributed
                )
                top_5_acc = tensor_to_python_float(
                    top_5_acc, is_distributed=is_distributed
                )
                top1_dict[target_k] = top_1_acc
                top5_dict[target_k] = top_5_acc
        return top1_dict, top5_dict
    else:
        logger.error("Metric monitor supports Tensor or Dict of Tensors")


def gather_iou_metrics(
    prediction: Union[Tensor, Dict], target: Tensor, is_distributed: bool
) -> Union[Tuple[Number, Number], Tuple[Dict[str, Number], Dict[str, Number]]]:
    """
    This function gathers intersection and union metrics from different processes and converts to float.
    """
    if isinstance(prediction, Tensor) and isinstance(target, Tensor):
        inter, union = compute_miou_batch(prediction=prediction, target=target)
        inter = tensor_to_python_float(inter, is_distributed=is_distributed)
        union = tensor_to_python_float(union, is_distributed=is_distributed)
        return inter, union
    # elif isinstance(prediction, Dict):
    #    logger.error("IOU metrics are not supported for a dictionary of predictions")
    # We will revisit it later, as per the use case.

    # inter_dict = {}
    # union_dict = {}
    # for k, v in prediction.items():
    #     inter, union = compute_miou_batch(prediction=v, target=target)
    #     inter = tensor_to_python_float(inter, is_distributed=is_distributed)
    #     union = tensor_to_python_float(union, is_distributed=is_distributed)
    #     inter_dict[k] = inter
    #     union_dict[k] = union
    # return inter_dict, union_dict
    else:
        logger.error("Metric monitor supports Tensor only for IoU")


def gather_psnr_metrics(
    prediction: Union[Tensor, Dict], target: Union[Tensor, Dict], is_distributed: bool
) -> Union[Number, Dict[str, Number]]:
    """
    This function gathers psnr scores from different processes and converts to float.
    """
    # We have four combinations between prediction and target types:
    # 1. (Tensor, Tensor)
    # 2. (Dict, Tensor)
    # 3. (Dict, Dict)
    # 4. (Tensor, Dict) --> This combination is rare

    if isinstance(prediction, Tensor) and isinstance(target, Tensor):
        if prediction.numel() != target.numel():
            logger.error(
                "Prediction and target have different number of elements."
                "Got: Prediction={} and target={}".format(
                    prediction.shape, target.shape
                )
            )
        psnr = compute_psnr(prediction=prediction, target=target)
        psnr = tensor_to_python_float(psnr, is_distributed=is_distributed)
        return psnr
    elif isinstance(prediction, Dict) and isinstance(target, Tensor):
        psnr_dict = {}
        for pred_k, pred_v in prediction.items():
            # only compute PSNR where prediction size and target sizes are the same
            if isinstance(pred_v, Tensor) and (pred_v.numel() == target.numel()):
                psnr = compute_psnr(prediction=pred_v, target=target)
                psnr = tensor_to_python_float(psnr, is_distributed=is_distributed)
                psnr_dict[pred_k] = psnr
        return psnr_dict
    elif isinstance(prediction, Dict) and isinstance(target, Dict):
        # prediction and target dictionaries should have intersecting keys
        prediction_keys = prediction.keys()
        target_keys = target.keys()

        intersection_keys = list(set(prediction_keys).intersection(target_keys))
        if len(intersection_keys) == 0:
            logger.error(
                "The keys in prediction and target are different. "
                " Got: Prediction keys={} and Target keys={}".format(
                    prediction_keys, target_keys
                )
            )

        psnr_dict = {}
        for pred_k in intersection_keys:
            pred_v = prediction[pred_k]
            target_v = target[pred_k]
            # only compute PSNR where prediction size and target sizes are the same
            if (
                isinstance(pred_v, Tensor)
                and isinstance(target_v, Tensor)
                and (pred_v.numel() == target_v.numel())
            ):
                psnr = compute_psnr(prediction=pred_v, target=target_v)
                psnr = tensor_to_python_float(psnr, is_distributed=is_distributed)
                psnr_dict[pred_k] = psnr
        return psnr_dict
    elif isinstance(prediction, Tensor) and isinstance(target, Dict):
        psnr_dict = {}
        for target_k, target_v in target.items():
            # only compute PSNR where prediction size and target sizes are the same
            if isinstance(target_v, Tensor) and (
                prediction.numel() == target_v.numel()
            ):
                psnr = compute_psnr(prediction=prediction, target=target_v)
                psnr = tensor_to_python_float(psnr, is_distributed=is_distributed)
                psnr_dict[target_k] = psnr
        return psnr_dict
    else:
        logger.error("Metric monitor supports Tensor or Dict of Tensors")

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#


def parse_validation_metric_names(opts):
    """
    This function contains common command-line parsing logic for validation metrics
    """
    metric_names = getattr(opts, "stats.val", ["loss"])
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    assert isinstance(
        metric_names, list
    ), "Type of metric names should be list. Got: {}".format(type(metric_names))

    if "loss" not in metric_names:
        metric_names.append("loss")

    ckpt_metric_str = getattr(opts, "stats.checkpoint_metric", "loss")
    ckpt_metric_arr = ckpt_metric_str.split(".")
    ckpt_metric = ckpt_metric_arr[0]
    if len(ckpt_metric_arr) == 1:
        ckpt_submetric_name = None
    else:
        ckpt_submetric_name = ckpt_metric_arr[-1]

    ckpt_metric = ckpt_metric
    ckpt_submetric = ckpt_submetric_name
    if ckpt_metric is None:
        # if checkpoint metric is not specified, then use loss
        ckpt_metric = "loss"

    assert (
        ckpt_metric in metric_names
    ), "Checkpoint metric should be part of metric names. Metric names: {}, Checkpoint metric: {}".format(
        metric_names, ckpt_metric
    )
    ckpt_metric = ckpt_metric.lower()

    return metric_names, ckpt_metric, ckpt_submetric

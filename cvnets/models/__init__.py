#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
from utils import logger

from .segmentation import arguments_segmentation, build_segmentation_model
from .classification import arguments_classification, build_classification_model
from .detection import arguments_detection, build_detection_model

SUPPORTED_TASKS = ["segmentation", "classification", "detection"]


def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    # classification network
    parser = arguments_classification(parser=parser)

    # detection network
    parser = arguments_detection(parser=parser)

    # segmentation network
    parser = arguments_segmentation(parser=parser)

    return parser


def get_model(opts):
    dataset_category = getattr(opts, "dataset.category", "classification")
    model = None
    if dataset_category == "classification":
        model = build_classification_model(opts=opts)
    elif dataset_category == "segmentation":
        model = build_segmentation_model(opts=opts)
    elif dataset_category == "detection":
        model = build_detection_model(opts=opts)
    else:
        task_str = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(dataset_category)
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        logger.error(task_str)

    return model
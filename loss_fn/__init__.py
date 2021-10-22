#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from .base_criteria import BaseCriteria
import os
import importlib
from utils import logger
import argparse

LOSS_REGISTRY = {}


def register_loss_fn(name):
    def register_loss_fn_class(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss function ({})".format(name))

        if not issubclass(cls, BaseCriteria):
            raise ValueError(
                "Criteria ({}: {}) must extend BaseCriteria".format(name, cls.__name__)
            )

        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_fn_class


def build_loss_fn(opts):
    loss_fn_category = getattr(opts, "loss.category", "classification").lower()
    loss_fn = None
    if loss_fn_category in LOSS_REGISTRY:
        loss_fn = LOSS_REGISTRY[loss_fn_category](opts)
    else:
        temp_list = list(LOSS_REGISTRY.keys())
        temp_str = "Loss function ({}) not yet supported. \n Supported loss functions are:".format(loss_fn_category)
        for i, m_name in enumerate(temp_list):
            temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(temp_str)

    return loss_fn


def general_loss_fn_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Loss function arguments", description="Loss function arguments")

    group.add_argument("--loss.category", type=str, default="classification",
                       help="Loss function category (classification,segmentation)")
    group.add_argument("--loss.ignore-idx", type=int, default=-1, help="Ignore idx in loss function")

    return parser


def arguments_loss_fn(parser: argparse.ArgumentParser):
    parser = general_loss_fn_args(parser=parser)

    # add loss function specific arguments
    for k, v in LOSS_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


# automatically import the loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        loss_fn_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("loss_fn." + loss_fn_name)

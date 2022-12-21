#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import argparse
import importlib

from utils import logger

SUPPORTED_DISTILL_LOSS_FNS = []
DISTILL_LOSS_FN_REGISTRY = {}


def register_distillation_loss_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_DISTILL_LOSS_FNS:
            raise ValueError(
                "Cannot register duplicate distillation loss function ({})".format(name)
            )
        SUPPORTED_DISTILL_LOSS_FNS.append(name)
        DISTILL_LOSS_FN_REGISTRY[name] = fn
        return fn

    return register_fn


def arguments_distill_loss_fn(parser: argparse.ArgumentParser):
    # add loss function specific arguments
    for k, v in DISTILL_LOSS_FN_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_loss_fn_str(loss_fn_name):
    supp_str = (
        "Loss function ({}) is not yet supported. \n Supported functions are:".format(
            loss_fn_name
        )
    )
    for i, fn_name in enumerate(SUPPORTED_DISTILL_LOSS_FNS):
        supp_str += "{} \t".format(fn_name)
    logger.error(supp_str)


def get_distillation_loss(opts, *args, **kwargs):
    loss_fn_name = getattr(opts, "loss.distillation.name", None)

    if loss_fn_name in SUPPORTED_DISTILL_LOSS_FNS:
        return DISTILL_LOSS_FN_REGISTRY[loss_fn_name](opts, *args, **kwargs)
    else:
        supported_loss_fn_str(loss_fn_name)
        return None


# automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("loss_fn.distillation_loss_fns." + model_name)

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import importlib
import os
import argparse

from utils import logger

from ..base_criteria import BaseCriteria

SUPPORTED_MULTI_MODAL_IMG_TEXT_LOSS_FNS = []
MULTI_MODAL_IMG_TEXT_LOSS_FN_REGISTRY = {}


def register_multi_modal_img_text_loss_fns(name):
    def register_fn(cls):
        if name in SUPPORTED_MULTI_MODAL_IMG_TEXT_LOSS_FNS:
            raise ValueError(
                "Cannot register duplicate multi-modal image-text loss function ({})".format(
                    name
                )
            )

        if not issubclass(cls, BaseCriteria):
            raise ValueError(
                "Loss function ({}: {}) must extend BaseCriteria".format(
                    name, cls.__name__
                )
            )

        MULTI_MODAL_IMG_TEXT_LOSS_FN_REGISTRY[name] = cls
        SUPPORTED_MULTI_MODAL_IMG_TEXT_LOSS_FNS.append(name)
        return cls

    return register_fn


def arguments_multi_modal_img_text_loss_fn(parser: argparse.ArgumentParser):
    # add loss function specific arguments
    for k, v in MULTI_MODAL_IMG_TEXT_LOSS_FN_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_loss_fn_str(loss_fn_name):
    supp_str = (
        "Loss function ({}) is not yet supported. \n Supported functions are:".format(
            loss_fn_name
        )
    )
    for i, fn_name in enumerate(SUPPORTED_MULTI_MODAL_IMG_TEXT_LOSS_FNS):
        supp_str += "{} \t".format(fn_name)
    logger.error(supp_str)


def get_multi_modal_img_text_loss(opts, *args, **kwargs):
    loss_name = getattr(opts, "loss.multi_modal_image_text.name", None)

    if loss_name in SUPPORTED_MULTI_MODAL_IMG_TEXT_LOSS_FNS:
        return MULTI_MODAL_IMG_TEXT_LOSS_FN_REGISTRY[loss_name](opts, *args, **kwargs)
    else:
        supported_loss_fn_str(loss_name)


# automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        loss_fn_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "loss_fn.multi_modal_img_text_loss_fns." + loss_fn_name
        )

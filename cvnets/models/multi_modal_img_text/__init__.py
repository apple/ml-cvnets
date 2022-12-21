#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
from typing import Optional

from utils.download_utils import get_local_path
from utils import logger
from utils.common_utils import check_frozen_norm_layer
from utils.ddp_utils import is_master

from .. import register_task_arguments, register_tasks
from .base_multi_modal_img_text import BaseMultiModalImageText
from ...misc.common import load_pretrained_model


MULTI_MODAL_IMAGE_TEXT_REGISTRY = {}


def register_multi_modal_image_text(name):
    # register the multi_modal_image_text class
    def register_multi_modal_image_text_class(cls):
        if name in MULTI_MODAL_IMAGE_TEXT_REGISTRY:
            raise ValueError(
                "Cannot register duplicate multi-modal-image-text class ({})".format(
                    name
                )
            )

        if not issubclass(cls, BaseMultiModalImageText):
            raise ValueError(
                "Multi-modal image text class ({}: {}) must extend BaseMultiModalImageText".format(
                    name, cls.__name__
                )
            )

        MULTI_MODAL_IMAGE_TEXT_REGISTRY[name] = cls
        return cls

    return register_multi_modal_image_text_class


@register_task_arguments(name="multi_modal_img_text")
def arguments_multi_modal_image_text(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    # add arguments for multi_modal_image_text
    parser = BaseMultiModalImageText.add_arguments(parser)

    # add model specific arguments
    for k, v in MULTI_MODAL_IMAGE_TEXT_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_model_str(model_name: Optional[str] = None) -> None:
    """Helper utility to print supported model names in case specified model_name
    is not part of the implemented models.
    """
    supp_list = list(MULTI_MODAL_IMAGE_TEXT_REGISTRY.keys())
    if model_name is None:
        supp_str = "Model name can't be None. \n Supported models are:"
    else:
        supp_str = "Model ({}) is not yet supported. \n Supported models are:".format(
            model_name
        )
    for t_name in supp_list:
        supp_str += "\n\t{}".format(t_name)
    logger.error(supp_str + "\n")


@register_tasks(name="multi_modal_img_text")
def build_multi_modal_image_text_model(
    opts, *args, **kwargs
) -> BaseMultiModalImageText:
    """Helper function to build the multi-modal image-text model"""
    model_name = getattr(opts, "model.multi_modal_image_text.name", None)
    if model_name is None:
        supported_model_str(model_name)

    model = None
    if model_name in list(MULTI_MODAL_IMAGE_TEXT_REGISTRY.keys()):
        model = MULTI_MODAL_IMAGE_TEXT_REGISTRY[model_name](opts, *args, **kwargs)
    else:
        supported_model_str(model_name)

    is_master_node = is_master(opts)
    pretrained = getattr(opts, "model.multi_modal_image_text.pretrained", None)
    if pretrained is not None:
        pretrained = get_local_path(opts, path=pretrained)
        model = load_pretrained_model(model=model, wt_loc=pretrained, opts=opts)

    freeze_norm_layers = getattr(
        opts, "model.multi_modal_image_text.freeze_batch_norm", False
    )
    if freeze_norm_layers:
        model.freeze_norm_layers()
        frozen_state, count_norm = check_frozen_norm_layer(model)
        if count_norm > 0 and frozen_state and is_master_node:
            logger.error(
                "Something is wrong while freezing normalization layers. Please check"
            )

        if is_master_node:
            logger.log("Normalization layers are frozen")

    return model


# automatically import the models
model_dir = os.path.dirname(__file__)

for file in os.listdir(model_dir):
    path = os.path.join(model_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "cvnets.models.multi_modal_img_text." + model_name
        )

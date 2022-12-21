#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
from typing import Optional

from utils import logger

from .base_image_projection import BaseImageProjectionHead


IMAGE_PROJECTION_HEAD_REGISTRY = {}


def register_image_projection_head(name):
    # register the image projection head class
    def register_image_projection_head_class(cls):
        if name in IMAGE_PROJECTION_HEAD_REGISTRY:
            raise ValueError(
                "Cannot register duplicate image projection layer class ({})".format(
                    name
                )
            )

        if not issubclass(cls, BaseImageProjectionHead):
            raise ValueError(
                "Image projection layer class ({}: {}) must extend BaseImageProjection".format(
                    name, cls.__name__
                )
            )

        IMAGE_PROJECTION_HEAD_REGISTRY[name] = cls
        return cls

    return register_image_projection_head_class


def arguments_image_projection_head(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    # add arguments for base image projection layer
    parser = BaseImageProjectionHead.add_arguments(parser)

    # add class specific arguments
    for k, v in IMAGE_PROJECTION_HEAD_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_str(layer_name: Optional[str] = None) -> None:
    """Helper utility to print supported image projection heads."""
    supp_list = list(IMAGE_PROJECTION_HEAD_REGISTRY.keys())
    if layer_name is None:
        supp_str = "Image projection head name can't be None. \n Supported heads are:"
    else:
        supp_str = "Image projection head ({}) is not yet supported. \n Supported heads are:".format(
            layer_name
        )
    for t_name in supp_list:
        supp_str += "\n\t{}".format(t_name)
    logger.error(supp_str + "\n")


def build_image_projection_head(
    opts, in_dim: int, out_dim: int, *args, **kwargs
) -> BaseImageProjectionHead:
    """Helper function to build the text encoder"""
    projection_head_name = getattr(opts, "model.image_projection_head.name", None)
    if projection_head_name is None:
        supported_str(projection_head_name)

    if projection_head_name in list(IMAGE_PROJECTION_HEAD_REGISTRY.keys()):
        return IMAGE_PROJECTION_HEAD_REGISTRY[projection_head_name](
            opts, in_dim, out_dim, *args, **kwargs
        )
    else:
        supported_str(projection_head_name)


# automatically import the image projection heads
image_projection_head_dir = os.path.dirname(__file__)

for file in os.listdir(image_projection_head_dir):
    path = os.path.join(image_projection_head_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        proj_head_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "cvnets.image_projection_layers." + proj_head_name
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
from typing import Optional

from utils import logger

from .base_text_encoder import BaseTextEncoder


TEXT_ENCODER_REGISTRY = {}


def register_text_encoder(name):
    # register the text_encoder class
    def register_text_encoder_class(cls):
        if name in TEXT_ENCODER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate text_encoder class ({})".format(name)
            )

        if not issubclass(cls, BaseTextEncoder):
            raise ValueError(
                "Text encoder class ({}: {}) must extend BaseTextEncoder".format(
                    name, cls.__name__
                )
            )

        TEXT_ENCODER_REGISTRY[name] = cls
        return cls

    return register_text_encoder_class


def arguments_text_encoder(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # add arguments for text_encoder
    parser = BaseTextEncoder.add_arguments(parser)

    # add augmentation specific arguments
    for k, v in TEXT_ENCODER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_text_encoder_str(text_encoder_name: Optional[str] = None) -> None:
    """Helper utility to print supported text_encoder names in case specified text_encoder
    name is not part of the implemented text encoders.
    """
    supp_list = list(TEXT_ENCODER_REGISTRY.keys())
    if text_encoder_name is None:
        supp_str = "Text encoder name can't be None. \n Supported text encoders are:"
    else:
        supp_str = "Text encoder ({}) is not yet supported. \n Supported text encoders are:".format(
            text_encoder_name
        )
    for t_name in supp_list:
        supp_str += "\n\t{}".format(t_name)
    logger.error(supp_str + "\n")


def build_text_encoder(opts, projection_dim: int, *args, **kwargs) -> BaseTextEncoder:
    """Helper function to build the text encoder"""
    text_encoder_name = getattr(opts, "model.text.name", None)
    if text_encoder_name is None:
        supported_text_encoder_str(text_encoder_name)

    if text_encoder_name in list(TEXT_ENCODER_REGISTRY.keys()):
        return TEXT_ENCODER_REGISTRY[text_encoder_name](
            opts, projection_dim, *args, **kwargs
        )
    else:
        supported_text_encoder_str(text_encoder_name)


# automatically import the text encoders
text_encoder_dir = os.path.dirname(__file__)

for file in os.listdir(text_encoder_dir):
    path = os.path.join(text_encoder_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        text_encoder_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.text_encoders." + text_encoder_name)

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
from typing import Optional

from utils import logger

from .base_tokenizer import BaseTokenizer


TOKENIZER_REGISTRY = {}


def register_tokenizer(name):
    # register the text_tokenizer class
    def register_tokenizer_class(cls):
        if name in TOKENIZER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate text_tokenizer class ({})".format(name)
            )

        if not issubclass(cls, BaseTokenizer):
            raise ValueError(
                "Tokenizer ({}: {}) must extend BaseTokenizer".format(
                    name, cls.__name__
                )
            )

        TOKENIZER_REGISTRY[name] = cls
        return cls

    return register_tokenizer_class


def arguments_tokenizer(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # add arguments for text_tokenizer
    parser = BaseTokenizer.add_arguments(parser)

    # add augmentation specific arguments
    for k, v in TOKENIZER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def supported_tokenizer_str(tokenizer_name: Optional[str] = None) -> None:
    """Helper utility to print supported text_tokenizer names in case specified text_tokenizer
    name is not part of the implemented tokenizers.
    """
    supp_list = list(TOKENIZER_REGISTRY.keys())
    if tokenizer_name is None:
        supp_str = "Tokenizer name can't be None. \n Supported tokenizers are:"
    else:
        supp_str = (
            "Tokenizer ({}) is not yet supported. \n Supported tokenizers are:".format(
                tokenizer_name
            )
        )
    for t_name in supp_list:
        supp_str += "\n\t{}".format(t_name)
    logger.error(supp_str + "\n")


def build_tokenizer(opts, *args, **kwargs) -> BaseTokenizer:
    """Helper function to build the text_tokenizer"""
    tokenizer_name = getattr(opts, "text_tokenizer.name", None)
    if tokenizer_name is None:
        supported_tokenizer_str(tokenizer_name)

    if tokenizer_name in list(TOKENIZER_REGISTRY.keys()):
        return TOKENIZER_REGISTRY[tokenizer_name](opts, *args, **kwargs)
    else:
        supported_tokenizer_str(tokenizer_name)


# automatically import the tokenizers
tokenizer_dir = os.path.dirname(__file__)

for file in os.listdir(tokenizer_dir):
    path = os.path.join(tokenizer_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        tokenizer_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("data.text_tokenizer." + tokenizer_name)

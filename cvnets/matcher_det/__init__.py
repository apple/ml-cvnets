#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import os
import importlib

from utils import logger
from utils.ddp_utils import is_master

from .base_matcher import BaseMatcher

# register BOX Matcher
MATCHER_REGISTRY = {}


def register_matcher(name):
    def register_class(cls):
        if name in MATCHER_REGISTRY:
            raise ValueError("Cannot register duplicate matcher ({})".format(name))

        if not issubclass(cls, BaseMatcher):
            raise ValueError(
                "Matcher ({}: {}) must extend BaseMatcher".format(name, cls.__name__)
            )

        MATCHER_REGISTRY[name] = cls
        return cls

    return register_class


def arguments_box_matcher(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Matcher", "Matcher")
    group.add_argument(
        "--matcher.name",
        type=str,
        help="Name of the matcher. Matcher matches anchors with GT box coordinates",
    )

    # add segmentation specific arguments
    for k, v in MATCHER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


def build_matcher(opts, *args, **kwargs):
    matcher_name = getattr(opts, "matcher.name", None)
    matcher = None
    if matcher_name in MATCHER_REGISTRY:
        matcher = MATCHER_REGISTRY[matcher_name](opts, *args, **kwargs)
    else:
        supported_matchers = list(MATCHER_REGISTRY.keys())
        supp_matcher_str = "Got {} as matcher. Supported matchers are:".format(
            matcher_name
        )
        for i, m_name in enumerate(supported_matchers):
            supp_matcher_str += "\n\t {}: {}".format(i, logger.color_text(m_name))

        if is_master(opts):
            logger.error(supp_matcher_str)
    return matcher


# automatically import the matchers
matcher_dir = os.path.dirname(__file__)
for file in os.listdir(matcher_dir):
    path = os.path.join(matcher_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        matcher_py = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.matcher_det." + matcher_py)

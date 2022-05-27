#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import os
import importlib

from utils import logger
from utils.ddp_utils import is_master

from .base_anchor_generator import BaseAnchorGenerator

# register anchor generator
ANCHOR_GEN_REGISTRY = {}


def register_anchor_generator(name):
    """Register anchor generators for object detection"""

    def register_class(cls):
        if name in ANCHOR_GEN_REGISTRY:
            raise ValueError(
                "Cannot register duplicate anchor generator ({})".format(name)
            )

        if not issubclass(cls, BaseAnchorGenerator):
            raise ValueError(
                "Anchor generator ({}: {}) must extend BaseAnchorGenerator".format(
                    name, cls.__name__
                )
            )

        ANCHOR_GEN_REGISTRY[name] = cls
        return cls

    return register_class


def arguments_anchor_gen(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments related to anchor generator for object detection"""
    group = parser.add_argument_group("Anchor generator", "Anchor generator")
    group.add_argument(
        "--anchor-generator.name", type=str, help="Name of the anchor generator"
    )

    for k, v in ANCHOR_GEN_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


def build_anchor_generator(opts, *args, **kwargs):
    """Build anchor generator for object detection"""
    anchor_gen_name = getattr(opts, "anchor_generator.name", None)
    anchor_gen = None
    if anchor_gen_name in ANCHOR_GEN_REGISTRY:
        anchor_gen = ANCHOR_GEN_REGISTRY[anchor_gen_name](opts, *args, **kwargs)
    else:
        supported_anchor_gens = list(ANCHOR_GEN_REGISTRY.keys())
        supp_anchor_gen_str = (
            "Got {} as anchor generator. Supported anchor generators are:".format(
                anchor_gen_name
            )
        )
        for i, m_name in enumerate(supported_anchor_gens):
            supp_anchor_gen_str += "\n\t {}: {}".format(i, logger.color_text(m_name))

        if is_master(opts):
            logger.error(supp_anchor_gen_str)
    return anchor_gen


# automatically import the anchor generators
anchor_gen_dir = os.path.dirname(__file__)
for file in os.listdir(anchor_gen_dir):
    path = os.path.join(anchor_gen_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        anc_gen = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.anchor_generator." + anc_gen)

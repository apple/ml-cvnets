#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
from typing import Optional
from utils import logger
import argparse


from .base_sampler import BaseSamplerDDP, BaseSamplerDP

SAMPLER_REGISTRY = {}


def register_sampler(name):
    def register_sampler_class(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError("Cannot register duplicate sampler class ({})".format(name))

        if not (issubclass(cls, BaseSamplerDDP) or issubclass(cls, BaseSamplerDP)):
            raise ValueError(
                "Sampler ({}: {}) must extend BaseSamplerDDP or BaseSamplerDP".format(name, cls.__name__)
            )

        SAMPLER_REGISTRY[name] = cls
        return cls

    return register_sampler_class


def build_sampler(opts, n_data_samples: int, is_training: Optional[bool] = False):
    sampler_name = getattr(opts, "sampler.name", "variable_batch_sampler")
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    if is_distributed and sampler_name.split('_')[-1] != "ddp":
        sampler_name = sampler_name + "_ddp"

    sampler = None
    if sampler_name in SAMPLER_REGISTRY:
        sampler = SAMPLER_REGISTRY[sampler_name](opts, n_data_samples=n_data_samples, is_training=is_training)
    else:
        supp_list = list(SAMPLER_REGISTRY.keys())
        supp_str = "Sampler ({}) not yet supported. \n Supported optimizers are:".format(sampler_name)
        for i, m_name in enumerate(supp_list):
            supp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_str)

    return sampler


def sampler_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--sampler.name', type=str, default="batch_sampler", help="Name of the sampler")

    return parser


def arguments_sampler(parser: argparse.ArgumentParser):
    parser = sampler_common_args(parser=parser)

    # add classification specific arguments
    for k, v in SAMPLER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


# automatically import the samplers
sampler_dir = os.path.dirname(__file__)
for file in os.listdir(sampler_dir):
    path = os.path.join(sampler_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        sampler_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("data.sampler." + sampler_name)

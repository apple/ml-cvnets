#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
import glob

from utils.ddp_utils import is_master
from utils import logger

from .base_zero_shot import BaseZeroShotDataset


ZERO_SHOT_DATASET_REGISTRY = {}

SEPARATOR = ":"


def register_zero_shot_dataset(name):
    """Helper function to register zero-shot datasets"""

    def register_zero_shot_dataset_class(cls):
        if name in ZERO_SHOT_DATASET_REGISTRY:
            raise ValueError(
                "Cannot register duplicate zero-shot dataset class ({})".format(name)
            )

        if not issubclass(cls, BaseZeroShotDataset):
            raise ValueError(
                "Zero shot dataset ({}: {}) must extend BaseZeroShotDataset".format(
                    name, cls.__name__
                )
            )

        ZERO_SHOT_DATASET_REGISTRY[name] = cls
        return cls

    return register_zero_shot_dataset_class


def supported_zero_shot_dataset_str(dataset_name) -> None:
    """Helper function to print error message in case zero shot dataset is not available"""

    supp_list = list(ZERO_SHOT_DATASET_REGISTRY.keys())
    supp_str = "Zero shot dataset ({}) is not yet supported. \n Supported datasets are:".format(
        dataset_name
    )
    for i, d_name in enumerate(supp_list):
        supp_str += "\n\t\t{}: {}".format(i, d_name)
    logger.error(supp_str + "\n")


def arguments_zero_shot_dataset(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Helper function to get zero-shot dataset arguments"""

    parser = BaseZeroShotDataset.add_arguments(parser=parser)

    # add dataset specific arguments
    for k, v in ZERO_SHOT_DATASET_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def build_zero_shot_dataset(opts, *args, **kwargs):
    """Helper function to build the zero shot datasets"""
    zero_shot_dataset_name = getattr(
        opts, "dataset.multi_modal_img_text.zero_shot.name", None
    )

    if zero_shot_dataset_name in list(ZERO_SHOT_DATASET_REGISTRY.keys()):
        return ZERO_SHOT_DATASET_REGISTRY[zero_shot_dataset_name](opts, *args, **kwargs)
    else:
        supported_zero_shot_dataset_str(zero_shot_dataset_name)


# automatically import zero-shot datasets
dataset_dir = os.path.dirname(__file__)

for file in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        zs_dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "data.datasets.multi_modal_img_text.zero_shot." + zs_dataset_name
        )

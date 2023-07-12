#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import os

from data.datasets.multi_modal_img_text.zero_shot.base_zero_shot import (
    BaseZeroShotDataset,
)
from utils.registry import Registry

ZERO_SHOT_DATASET_REGISTRY = Registry(
    registry_name="zero_shot_datasets",
    base_class=BaseZeroShotDataset,
    lazy_load_dirs=["data/datasets/multi_modal_img_text/zero_shot"],
    internal_dirs=["internal", "internal/projects/*"],
)


def arguments_zero_shot_dataset(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Helper function to get zero-shot dataset arguments"""
    parser = BaseZeroShotDataset.add_arguments(parser=parser)
    parser = ZERO_SHOT_DATASET_REGISTRY.all_arguments(parser)
    return parser


def build_zero_shot_dataset(opts, *args, **kwargs) -> BaseZeroShotDataset:
    """Helper function to build the zero shot datasets"""
    zero_shot_dataset_name = getattr(
        opts, "dataset.multi_modal_img_text.zero_shot.name"
    )
    return ZERO_SHOT_DATASET_REGISTRY[zero_shot_dataset_name](opts, *args, **kwargs)

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

from .base_transforms import BaseTransformation

SUPPORTED_AUG_CATEGORIES = []
AUGMENTAION_REGISTRY = {}


def register_transformations(name, type):
    def register_transformation_class(cls):
        if name in AUGMENTAION_REGISTRY:
            raise ValueError("Cannot register duplicate transformation class ({})".format(name))

        if not issubclass(cls, BaseTransformation):
            raise ValueError(
                "Transformation ({}: {}) must extend BaseTransformation".format(name, cls.__name__)
            )

        AUGMENTAION_REGISTRY[name + "_" + type] = cls
        return cls
    return register_transformation_class


def arguments_augmentation(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    # add augmentation specific arguments
    for k, v in AUGMENTAION_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


# automatically import the augmentations
transform_dir = os.path.dirname(__file__)

for file in os.listdir(transform_dir):
    path = os.path.join(transform_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        transform_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("data.transforms." + transform_name)
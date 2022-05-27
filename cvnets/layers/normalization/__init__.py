#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

SUPPORTED_NORM_FNS = []


def register_norm_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(name)
            )
        SUPPORTED_NORM_FNS.append(name)
        return fn

    return register_fn


# automatically import different normalization layers
norm_dir = os.path.dirname(__file__)
for file in os.listdir(norm_dir):
    path = os.path.join(norm_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.layers.normalization." + model_name)


def arguments_norm_layers(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Normalization layers", description="Normalization layers"
    )

    group.add_argument(
        "--model.normalization.name",
        default=None,
        type=str,
        help="Normalization layer. Defaults to None",
    )
    group.add_argument(
        "--model.normalization.groups",
        default=1,
        type=str,
        help="Number of groups in group normalization layer. Defaults to 1.",
    )
    group.add_argument(
        "--model.normalization.momentum",
        default=0.1,
        type=float,
        help="Momentum in normalization layers. Defaults to 0.1",
    )

    # Adjust momentum in batch norm layers
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.enable",
        action="store_true",
        help="Adjust momentum in batch normalization layers",
    )
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.anneal-type",
        default="cosine",
        type=str,
        help="Method for annealing momentum in Batch normalization layer",
    )
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.final-momentum-value",
        default=1e-6,
        type=float,
        help="Min. momentum in batch normalization layer",
    )

    return parser


# import here to avoid circular loop
from cvnets.layers.normalization.batch_norm import BatchNorm2d, BatchNorm1d, BatchNorm3d
from cvnets.layers.normalization.group_norm import GroupNorm
from cvnets.layers.normalization.instance_norm import InstanceNorm1d, InstanceNorm2d
from cvnets.layers.normalization.sync_batch_norm import SyncBatchNorm
from cvnets.layers.normalization.layer_norm import LayerNorm, LayerNorm2D


__all__ = [
    "BatchNorm3d",
    "BatchNorm2d",
    "BatchNorm1d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "SyncBatchNorm",
    "LayerNorm",
    "LayerNorm2D",
]

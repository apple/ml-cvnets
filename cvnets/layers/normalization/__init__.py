#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import os
import importlib
import argparse
from typing import Optional

from utils import logger

from ..identity import Identity

SUPPORTED_NORM_FNS = []
NORM_LAYER_REGISTRY = {}
NORM_LAYER_CLS = []


def register_norm_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(name)
            )
        SUPPORTED_NORM_FNS.append(name)
        NORM_LAYER_REGISTRY[name] = cls
        NORM_LAYER_CLS.append(cls)
        return cls

    return register_fn


def build_normalization_layer(
    opts,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument
    """
    norm_type = (
        getattr(opts, "model.normalization.name", "batch_norm")
        if norm_type is None
        else norm_type
    )
    num_groups = (
        getattr(opts, "model.normalization.groups", 1)
        if num_groups is None
        else num_groups
    )
    momentum = getattr(opts, "model.normalization.momentum", 0.1)
    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None

    if norm_type in NORM_LAYER_REGISTRY:
        if torch.cuda.device_count() < 1 and norm_type.find("sync_batch") > -1:
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    elif norm_type == "identity":
        norm_layer = Identity()
    else:
        logger.error(
            "Supported normalization layer arguments are: {}. Got: {}".format(
                SUPPORTED_NORM_FNS, norm_type
            )
        )
    return norm_layer


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

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

SUPPORTED_ACT_FNS = []


def register_act_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(
                "Cannot register duplicate activation function ({})".format(name)
            )
        SUPPORTED_ACT_FNS.append(name)
        return fn

    return register_fn


# automatically import different activation functions
act_dir = os.path.dirname(__file__)
for file in os.listdir(act_dir):
    path = os.path.join(act_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.layers.activation." + model_name)


def arguments_activation_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Non-linear functions", description="Non-linear functions"
    )

    group.add_argument(
        "--model.activation.name",
        default="relu",
        type=str,
        help="Non-linear function name",
    )
    group.add_argument(
        "--model.activation.inplace",
        action="store_true",
        help="Use non-linear functions inplace",
    )
    group.add_argument(
        "--model.activation.neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu function",
    )

    return parser


# import later to avoid circular loop
from .gelu import GELU
from .hard_sigmoid import Hardsigmoid
from .hard_swish import Hardswish
from .leaky_relu import LeakyReLU
from .prelu import PReLU
from .relu import ReLU
from .relu6 import ReLU6
from .sigmoid import Sigmoid
from .swish import Swish
from .tanh import Tanh


__all__ = [
    "GELU",
    "Hardsigmoid",
    "Hardswish",
    "LeakyReLU",
    "PReLU",
    "ReLU",
    "ReLU6",
    "Sigmoid",
    "Swish",
    "Tanh",
]

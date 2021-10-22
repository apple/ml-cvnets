#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib

import torch.nn

from utils import logger
import argparse

from .base_optim import BaseOptim

OPTIM_REGISTRY = {}


def register_optimizer(name: str):
    def register_optimizer_class(cls):
        if name in OPTIM_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))

        if not issubclass(cls, BaseOptim):
            raise ValueError(
                "Optimizer ({}: {}) must extend BaseOptim".format(name, cls.__name__)
            )

        OPTIM_REGISTRY[name] = cls
        return cls

    return register_optimizer_class


def build_optimizer(model: torch.nn.Module, opts) -> BaseOptim:
    optim_name = getattr(opts, "optim.name", "sgd").lower()
    optimizer = None
    weight_decay = getattr(opts, "optim.weight_decay", 0.0)
    no_decay_bn_filter_bias = getattr(opts, "optim.no_decay_bn_filter_bias", False)

    if hasattr(model, "module"):
        model_params, lr_mult = model.module.get_trainable_parameters(weight_decay=weight_decay,
                                                                      no_decay_bn_filter_bias=no_decay_bn_filter_bias)
    else:
        model_params, lr_mult = model.get_trainable_parameters(weight_decay=weight_decay,
                                                               no_decay_bn_filter_bias=no_decay_bn_filter_bias)
    setattr(opts, "optim.lr_multipliers", lr_mult)
    if optim_name in OPTIM_REGISTRY:
        optimizer = OPTIM_REGISTRY[optim_name](opts, model_params)
    else:
        supp_list = list(OPTIM_REGISTRY.keys())
        supp_str = "Optimizer ({}) not yet supported. \n Supported optimizers are:".format(optim_name)
        for i, m_name in enumerate(supp_list):
            supp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_str)

    return optimizer


def general_optim_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group('optimizer', 'Optimizer related arguments')
    group.add_argument('--optim.name', default='sgd', help='Which optimizer')
    group.add_argument('--optim.eps', type=float, default=1e-8, help='Optimizer eps')
    group.add_argument('--optim.weight-decay', default=4e-5, type=float, help='Weight decay')
    group.add_argument('--optim.no-decay-bn-filter-bias', action="store_true",
                       help="No weight decay in normalization layers and bias")
    return parser


def arguments_optimizer(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = general_optim_args(parser=parser)

    # add optim specific arguments
    for k, v in OPTIM_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


# automatically import the optimizers
optim_dir = os.path.dirname(__file__)
for file in os.listdir(optim_dir):
    path = os.path.join(optim_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        optim_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("optim." + optim_name)

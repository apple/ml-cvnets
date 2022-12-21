#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
from typing import List, Dict
import torch.nn
import argparse

from utils import logger

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


def check_trainable_parameters(model: torch.nn.Module, model_params: List) -> None:
    """Helper function to check if any model parameters w/ gradients are not part of model_params"""

    # get model parameter names
    model_trainable_params = []
    for p_name, param in model.named_parameters():
        if param.requires_grad:
            model_trainable_params.append(p_name)

    initialized_params = []
    for param_info in model_params:
        if not isinstance(param_info, Dict):
            logger.error(
                "Expected format is a Dict with three keys: params, weight_decay, param_names"
            )

        if not {"params", "weight_decay", "param_names"}.issubset(param_info.keys()):
            logger.error(
                "Parameter dict should have three keys: params, weight_decay, param_names"
            )

        param_names = param_info.pop("param_names")
        if isinstance(param_names, List):
            initialized_params.extend(param_names)
        elif isinstance(param_names, str):
            initialized_params.append(param_names)
        else:
            raise NotImplementedError

    uninitialized_params = set(model_trainable_params) ^ set(initialized_params)
    if len(uninitialized_params) > 0:
        logger.error(
            "Following parameters are defined in the model, but won't be part of optimizer. "
            "Please check get_trainable_parameters function. "
            "Use --optim.bypass-parameters-check flag to bypass this check. "
            "Parameter list = {}".format(uninitialized_params)
        )


def remove_param_name_key(model_params: List) -> None:
    """Helper function to remove param_names key from model_params"""
    for param_info in model_params:
        if not isinstance(param_info, Dict):
            logger.error(
                "Expected format is a Dict with three keys: params, weight_decay, param_names"
            )

        if not {"params", "weight_decay", "param_names"}.issubset(param_info.keys()):
            logger.error(
                "Parameter dict should have three keys: params, weight_decay, param_names"
            )

        param_info.pop("param_names")


def build_optimizer(model: torch.nn.Module, opts, *args, **kwargs) -> BaseOptim:
    optim_name = getattr(opts, "optim.name", "sgd").lower()
    optimizer = None
    weight_decay = getattr(opts, "optim.weight_decay", 0.0)
    no_decay_bn_filter_bias = getattr(opts, "optim.no_decay_bn_filter_bias", False)

    unwrapped_model = model.module if hasattr(model, "module") else model

    model_params, lr_mult = unwrapped_model.get_trainable_parameters(
        weight_decay=weight_decay,
        no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        *args,
        **kwargs
    )

    # check to ensure that all trainable model parameters are passed to the model
    if not getattr(opts, "optim.bypass_parameters_check", False):
        check_trainable_parameters(model=unwrapped_model, model_params=model_params)
    else:
        remove_param_name_key(model_params=model_params)

    setattr(opts, "optim.lr_multipliers", lr_mult)
    if optim_name in OPTIM_REGISTRY:
        optimizer = OPTIM_REGISTRY[optim_name](opts, model_params)
    else:
        supp_list = list(OPTIM_REGISTRY.keys())
        supp_str = (
            "Optimizer ({}) not yet supported. \n Supported optimizers are:".format(
                optim_name
            )
        )
        for i, m_name in enumerate(supp_list):
            supp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_str)

    return optimizer


def general_optim_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("optimizer", "Optimizer related arguments")
    group.add_argument("--optim.name", default="sgd", help="Which optimizer")
    group.add_argument("--optim.eps", type=float, default=1e-8, help="Optimizer eps")
    group.add_argument(
        "--optim.weight-decay", default=4e-5, type=float, help="Weight decay"
    )
    group.add_argument(
        "--optim.no-decay-bn-filter-bias",
        action="store_true",
        help="No weight decay in normalization layers and bias",
    )
    group.add_argument(
        "--optim.bypass-parameters-check",
        action="store_true",
        help="Bypass parameter check when creating optimizer",
    )
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

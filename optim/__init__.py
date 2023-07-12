#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, List

import torch.nn

from optim.base_optim import BaseOptim
from utils import logger
from utils.common_utils import unwrap_model_fn
from utils.registry import Registry

OPTIM_REGISTRY = Registry(
    registry_name="optimizer_registry",
    base_class=BaseOptim,
    lazy_load_dirs=["optim"],
    internal_dirs=["internal", "internal/projects/*"],
)


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
    """Helper function to build an optimizer

    Args:
        model: A model
        opts: command-line arguments

    Returns:
        An instance of BaseOptim
    """
    optim_name = getattr(opts, "optim.name")
    weight_decay = getattr(opts, "optim.weight_decay")
    no_decay_bn_filter_bias = getattr(opts, "optim.no_decay_bn_filter_bias")

    unwrapped_model = unwrap_model_fn(model)

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

    # set the learning rate multiplier for each parameter
    setattr(opts, "optim.lr_multipliers", lr_mult)

    return OPTIM_REGISTRY[optim_name](opts, model_params, *args, **kwargs)


def arguments_optimizer(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = BaseOptim.add_arguments(parser)
    parser = OPTIM_REGISTRY.all_arguments(parser)
    return parser

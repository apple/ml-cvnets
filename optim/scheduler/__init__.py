#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
from utils import logger
import argparse

from .base_scheduler import BaseLRScheduler

SCHEDULER_REGISTRY = {}


def register_scheduler(name: str):
    def register_scheduler_class(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError("Cannot register duplicate scheduler ({})".format(name))

        if not issubclass(cls, BaseLRScheduler):
            raise ValueError(
                "LR Scheduler ({}: {}) must extend BaseLRScheduler".format(name, cls.__name__)
            )

        SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_scheduler_class


def build_scheduler(opts) -> BaseLRScheduler:
    scheduler_name = getattr(opts, "scheduler.name", "cosine").lower()
    lr_scheduler = None
    if scheduler_name in SCHEDULER_REGISTRY:
        lr_scheduler = SCHEDULER_REGISTRY[scheduler_name](opts)
    else:
        supp_list = list(SCHEDULER_REGISTRY.keys())
        supp_str = "LR Scheduler ({}) not yet supported. \n Supported schedulers are:".format(scheduler_name)
        for i, m_name in enumerate(supp_list):
            supp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_str)

    return lr_scheduler


def general_lr_sch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="LR scheduler arguments", description="LR scheduler arguments")

    group.add_argument('--scheduler.name', type=str, default="cosine", help="LR scheduler name")
    group.add_argument('--scheduler.lr', type=float, default=0.1, help="Learning rate")
    group.add_argument('--scheduler.max-epochs', type=int, default=None, help="Max. epochs for training")
    group.add_argument('--scheduler.max-iterations', type=int, default=None, help="Max. iterations for training")
    group.add_argument('--scheduler.warmup-iterations', type=int, default=0, help="Warm-up iterations")
    group.add_argument('--scheduler.warmup-init-lr', type=float, default=1e-7, help="Warm-up init lr")
    group.add_argument('--scheduler.is-iteration-based', action="store_true", help="Is iteration type or epoch type")

    return parser


def arguments_scheduler(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = general_lr_sch_args(parser=parser)

    # add scheduler specific arguments
    for k, v in SCHEDULER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


# automatically import the LR schedulers
lr_sch_dir = os.path.dirname(__file__)
for file in os.listdir(lr_sch_dir):
    path = os.path.join(lr_sch_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        lr_sch_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("optim.scheduler." + lr_sch_name)

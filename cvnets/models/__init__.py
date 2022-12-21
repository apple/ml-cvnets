#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import importlib
import argparse
from utils import logger
import os


SUPPORTED_TASKS = []
TASK_REGISTRY = {}
TASK_ARG_REGISTRY = {}


def register_tasks(name):
    def register_task_class(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))

        TASK_REGISTRY[name] = cls
        SUPPORTED_TASKS.append(name)
        return cls

    return register_task_class


def register_task_arguments(name):
    def register_task_arg_fn(fn):
        if name in TASK_ARG_REGISTRY:
            raise ValueError(
                "Cannot register duplicate task arguments ({})".format(name)
            )

        TASK_ARG_REGISTRY[name] = fn
        return fn

    return register_task_arg_fn


def common_model_argumnets(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # load model scopes
    parser.add_argument(
        "--model.resume-exclude-scopes",
        type=str,
        default="",
        help="Comma-separated list of parameter scopes (regex strings) to exclude when loading a pre-trained model",
    )
    parser.add_argument(
        "--model.ignore-missing-scopes",
        type=str,
        default="",
        help="Comma-separated list of parameter scopes (regex strings) to ignore if they are missing from the pre-training model",
    )
    parser.add_argument(
        "--model.rename-scopes-map",
        type=list,
        default=None,
        help="A mapping from checkpoint variable names to match the existing model names."
        " The mapping is represented as a List[List[str]], e.g. [['before', 'after'], ['this', 'that']]."
        " Note: only loading from Yaml file is supported for this argument.",
    )
    return parser


def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # common arguments
    parser = common_model_argumnets(parser=parser)

    for k, v in TASK_ARG_REGISTRY.items():
        parser = v(parser)
    return parser


def get_model(opts, *args, **kwargs):
    dataset_category = getattr(opts, "dataset.category", None)
    if not dataset_category:
        task_str = "Please specify dataset.category. Supported categories are:"
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        logger.error(task_str)

    dataset_category = dataset_category.lower()

    if dataset_category in TASK_REGISTRY:
        return TASK_REGISTRY[dataset_category](opts, *args, **kwargs)
    else:
        task_str = (
            "Got {} as a task. Unfortunately, we do not support it yet."
            "\nSupported tasks are:".format(dataset_category)
        )
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
        logger.error(task_str)


# automatically import the tasks
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        task_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.models." + task_name)

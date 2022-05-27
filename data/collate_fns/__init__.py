#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

COLLATE_FN_REGISTRY = {}


def register_collate_fn(name):
    def register_collate_fn_method(f):
        if name in COLLATE_FN_REGISTRY:
            raise ValueError(
                "Cannot register duplicate collate function ({})".format(name)
            )
        COLLATE_FN_REGISTRY[name] = f
        return f

    return register_collate_fn_method


def arguments_collate_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Collate function arguments", description="Collate function arguments"
    )
    group.add_argument(
        "--dataset.collate-fn-name-train",
        type=str,
        default="default_collate_fn",
        help="Name of collate function",
    )
    group.add_argument(
        "--dataset.collate-fn-name-val",
        type=str,
        default="default_collate_fn",
        help="Name of collate function",
    )
    group.add_argument(
        "--dataset.collate-fn-name-eval",
        type=str,
        default=None,
        help="Name of collate function used for evaluation. "
        "Default is None, i.e., use PyTorch's inbuilt collate function",
    )
    return parser


def build_collate_fn(opts, *args, **kwargs):
    collate_fn_name_train = getattr(
        opts, "dataset.collate_fn_name_train", "default_collate_fn"
    )
    collate_fn_name_val = getattr(
        opts, "dataset.collate_fn_name_val", "default_collate_fn"
    )
    collate_fn_train = None
    if (
        collate_fn_name_train is not None
        and collate_fn_name_train in COLLATE_FN_REGISTRY
    ):
        collate_fn_train = COLLATE_FN_REGISTRY[collate_fn_name_train]

    collate_fn_val = None
    if collate_fn_name_val is None:
        collate_fn_val = collate_fn_name_train
    elif collate_fn_name_val is not None and collate_fn_name_val in COLLATE_FN_REGISTRY:
        collate_fn_val = COLLATE_FN_REGISTRY[collate_fn_name_val]

    return collate_fn_train, collate_fn_val


def build_eval_collate_fn(opts, *args, **kwargs):
    collate_fn_name_eval = getattr(opts, "dataset.collate_fn_name_eval", None)
    collate_fn_eval = None
    if collate_fn_name_eval is not None and collate_fn_name_eval in COLLATE_FN_REGISTRY:
        collate_fn_eval = COLLATE_FN_REGISTRY[collate_fn_name_eval]

    return collate_fn_eval


# automatically import the augmentations
collate_fn_dir = os.path.dirname(__file__)

for file in os.listdir(collate_fn_dir):
    path = os.path.join(collate_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        collate_fn_fname = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("data.collate_fns." + collate_fn_fname)

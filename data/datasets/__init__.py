#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
import glob

from utils.ddp_utils import is_master
from utils import logger

from .dataset_base import BaseImageDataset


SUPPORTED_TASKS = []
DATASET_REGISTRY = {}

SEPARATOR = ":"


def register_dataset(name, task):
    def register_dataset_class(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(
                "Cannot register duplicate dataset class ({})".format(name)
            )

        if not issubclass(cls, BaseImageDataset):
            raise ValueError(
                "Dataset ({}: {}) must extend BaseImageDataset".format(
                    name, cls.__name__
                )
            )

        DATASET_REGISTRY[name + SEPARATOR + task] = cls
        return cls

    return register_dataset_class


def supported_dataset_str(dataset_name, dataset_category):
    supp_list = list(DATASET_REGISTRY.keys())
    supp_str = "Dataset ({}) under task ({}) is not yet supported. \n Supported datasets are:".format(
        dataset_name, dataset_category
    )
    for t_name in SUPPORTED_TASKS:
        supp_str += "\n\t {}: ".format(logger.color_text(t_name))
        for i, m_name in enumerate(supp_list):
            d_name, t_name1 = m_name.split(SEPARATOR)
            if t_name == t_name1:
                supp_str += "{} \t".format(d_name)
    logger.error(supp_str)


def evaluation_datasets(opts):
    dataset_name = getattr(opts, "dataset.name", "imagenet")
    dataset_category = getattr(opts, "dataset.category", "classification")

    is_master_node = is_master(opts)

    name_dataset_task = dataset_name + SEPARATOR + dataset_category
    eval_dataset = None
    if name_dataset_task in DATASET_REGISTRY:
        eval_dataset = DATASET_REGISTRY[name_dataset_task](
            opts=opts, is_training=False, is_evaluation=True
        )
    else:
        supported_dataset_str(
            dataset_name=dataset_name, dataset_category=dataset_category
        )

    if is_master_node:
        logger.log("Evaluation dataset details: ")
        print("{}".format(eval_dataset))

    return eval_dataset


def train_val_datasets(opts):
    dataset_name = getattr(opts, "dataset.name", "imagenet")
    dataset_category = getattr(opts, "dataset.category", "classification")
    disable_val = getattr(opts, "dataset.disable_val", False)

    is_master_node = is_master(opts)

    name_dataset_task = dataset_name + SEPARATOR + dataset_category
    train_dataset = valid_dataset = None
    if name_dataset_task in DATASET_REGISTRY and not disable_val:
        train_dataset = DATASET_REGISTRY[name_dataset_task](opts=opts, is_training=True)
        valid_dataset = DATASET_REGISTRY[name_dataset_task](
            opts=opts, is_training=False
        )
    elif name_dataset_task in DATASET_REGISTRY and disable_val:
        train_dataset = DATASET_REGISTRY[name_dataset_task](opts=opts, is_training=True)
        valid_dataset = None
    else:
        supported_dataset_str(
            dataset_name=dataset_name, dataset_category=dataset_category
        )

    if is_master_node:
        logger.log("Training and validation dataset details: ")
        print("{}".format(train_dataset))
        print("{}".format(valid_dataset))
    return train_dataset, valid_dataset


def trove_dataset_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Trove arguments")
    group.add_argument(
        "--dataset.trove.enable", action="store_true", help="Use trove for data loading"
    )
    group.add_argument(
        "--dataset.trove.mount-path",
        type=str,
        default="/mnt/vision_datasets",
        help="Local path for mounting dataset",
    )
    group.add_argument(
        "--dataset.trove.disk-cache-dir",
        type=str,
        default="/mnt/trove_cache",
        help="Local path for caching dataset",
    )
    group.add_argument(
        "--dataset.trove.disk-cache-max-size-gb",
        type=int,
        default=0,
        help="Disk cache size in GB",
    )
    group.add_argument(
        "--dataset.trove.disk-cache-mount-size-gb",
        type=int,
        default=0,
        help="Disk cache mount size in GB.",
    )
    group.add_argument("--dataset.trove.uri", type=str, default=None, help="Trove URI")
    group.add_argument(
        "--dataset.trove.dir-train",
        type=str,
        default=None,
        help="Location of train dataset inside trove.",
    )
    group.add_argument(
        "--dataset.trove.dir-val",
        type=str,
        default=None,
        help="Location of validation dataset inside trove.",
    )
    return parser


def general_dataset_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Dataset", description="Arguments related to dataset"
    )
    group.add_argument(
        "--dataset.root-train",
        type=str,
        default="",
        help="Root location of train dataset",
    )
    group.add_argument(
        "--dataset.root-val",
        type=str,
        default="",
        help="Root location of valid dataset",
    )
    group.add_argument(
        "--dataset.root-test",
        type=str,
        default="",
        help="Root location of test dataset",
    )
    group.add_argument(
        "--dataset.disable-val", action="store_true", help="Disable validation"
    )

    group.add_argument(
        "--dataset.name", type=str, default="imagenet", help="Dataset name"
    )
    group.add_argument(
        "--dataset.category",
        type=str,
        default="classification",
        help="Dataset category (e.g., segmentation, classification)",
    )
    group.add_argument(
        "--dataset.train-batch-size0", default=128, type=int, help="Training batch size"
    )
    group.add_argument(
        "--dataset.val-batch-size0", default=1, type=int, help="Validation batch size"
    )
    group.add_argument(
        "--dataset.eval-batch-size0", default=1, type=int, help="Validation batch size"
    )
    group.add_argument(
        "--dataset.workers", default=-1, type=int, help="Number of data workers"
    )
    group.add_argument(
        "--dataset.dali-workers",
        default=-1,
        type=int,
        help="Number of data workers for dali",
    )
    group.add_argument(
        "--dataset.persistent-workers",
        action="store_true",
        help="Use same workers across all epochs in data loader",
    )
    group.add_argument(
        "--dataset.pin-memory",
        action="store_true",
        help="Use pin memory option in data loader",
    )
    group.add_argument(
        "--dataset.prefetch-factor",
        type=int,
        default=2,
        help="Number of samples loaded in advance by each data worker",
    )
    group.add_argument(
        "--dataset.img-dtype",
        type=str,
        choices=["float", "half", "float16"],
        default="float",
        help="Image datatype",
    )

    group.add_argument(
        "--dataset.cache-images-on-ram", action="store_true", help="Cache data on RAM"
    )
    group.add_argument(
        "--dataset.cache-limit",
        type=float,
        default=80.0,
        help="Max. memory to use in RAM.",
    )

    # sample efficient training
    group.add_argument(
        "--dataset.sample-efficient-training.enable",
        action="store_true",
        help="sample efficient training",
    )
    group.add_argument(
        "--dataset.sample-efficient-training.sample-confidence",
        type=float,
        default=0.5,
        help="Confidence for sample",
    )
    group.add_argument(
        "--dataset.sample-efficient-training.find-easy-samples-every-k-epochs",
        type=int,
        default=5,
        help="Find easy samples after every K epochs",
    )
    group.add_argument(
        "--dataset.sample-efficient-training.min-sample-frequency",
        type=int,
        default=5,
        help="Frequency that sample has been classified as easy for N number of times.",
    )

    group.add_argument(
        "--dataset.decode-data-on-gpu", action="store_true", help="Decode data on GPU"
    )
    group.add_argument(
        "--dataset.sampler-type",
        type=str,
        default="batch",
        help="Batch sampler or not.",
    )

    return parser


def arguments_dataset(parser: argparse.ArgumentParser):
    parser = general_dataset_args(parser=parser)
    parser = trove_dataset_args(parser=parser)

    # add dataset specific arguments
    for k, v in DATASET_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


# automatically import the datasets
dataset_dir = os.path.dirname(__file__)

# supported tasks (each folder in datasets is for a particular task)
for abs_dir_path in glob.glob("{}/*".format(dataset_dir)):
    if os.path.isdir(abs_dir_path):
        file_or_folder_name = os.path.basename(abs_dir_path).strip()
        if not file_or_folder_name.startswith(
            "_"
        ) and not file_or_folder_name.startswith("."):
            SUPPORTED_TASKS.append(file_or_folder_name)

for task in SUPPORTED_TASKS:
    task_path = os.path.join(dataset_dir, task)
    for file in os.listdir(task_path):
        path = os.path.join(task_path, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
            module = importlib.import_module(
                "data.datasets." + task + "." + dataset_name
            )

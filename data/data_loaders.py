#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from functools import partial
from typing import Mapping, Optional, Tuple, Union

from torch.utils.data.sampler import Sampler

from data.collate_fns import build_collate_fn, build_test_collate_fn
from data.datasets import BaseDataset, get_test_dataset, get_train_val_datasets
from data.loader.dataloader import CVNetsDataLoader
from data.sampler import build_sampler
from utils import logger
from utils.ddp_utils import is_master
from utils.tensor_utils import image_size_from_opts


def create_test_loader(opts: argparse.Namespace) -> CVNetsDataLoader:
    """Helper function to create and return a dataset loader for test dataset from command-line arguments"""
    test_dataset = get_test_dataset(opts)

    n_test_samples = get_num_data_samples_as_int_or_mapping(test_dataset)
    is_master_node = is_master(opts)

    # overwrite the validation argument
    setattr(
        opts,
        "dataset.val_batch_size0",
        getattr(opts, "dataset.eval_batch_size0"),
    )

    # we don't need variable batch sampler for evaluation
    sampler_name = getattr(opts, "sampler.name", "batch_sampler")
    crop_size_h, crop_size_w = image_size_from_opts(opts)
    if sampler_name.find("video") > -1 and sampler_name != "video_batch_sampler":
        clips_per_video = getattr(opts, "sampler.vbs.clips_per_video", 1)
        frames_per_clip = getattr(opts, "sampler.vbs.num_frames_per_clip", 8)
        setattr(opts, "sampler.name", "video_batch_sampler")
        setattr(opts, "sampler.bs.crop_size_width", crop_size_w)
        setattr(opts, "sampler.bs.crop_size_height", crop_size_h)
        setattr(opts, "sampler.bs.clips_per_video", clips_per_video)
        setattr(opts, "sampler.bs.num_frames_per_clip", frames_per_clip)
    elif sampler_name.find("var") > -1:
        setattr(opts, "sampler.name", "batch_sampler")
        setattr(opts, "sampler.bs.crop_size_width", crop_size_w)
        setattr(opts, "sampler.bs.crop_size_height", crop_size_h)

    test_sampler = build_sampler(
        opts=opts,
        n_data_samples=n_test_samples,
        is_training=False,
        get_item_metadata=test_dataset.get_item_metadata,
    )

    collate_fn_test = build_test_collate_fn(opts=opts)

    data_workers = getattr(opts, "dataset.workers", 1)
    persistent_workers = False
    pin_memory = False

    test_loader = CVNetsDataLoader(
        dataset=test_dataset,
        batch_size=1,
        batch_sampler=test_sampler,
        num_workers=data_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=partial(collate_fn_test, opts=opts)
        if collate_fn_test is not None
        else None,
    )

    if is_master_node:
        logger.log("Evaluation sampler details: ")
        print("{}".format(test_sampler))

    return test_loader


def create_train_val_loader(
    opts: argparse.Namespace,
) -> Tuple[CVNetsDataLoader, Optional[CVNetsDataLoader], Sampler]:
    """Helper function to create training and validation data loaders.

    Args:
        opts: Command-line arguments

    Returns:
        A tuple containing training data loader, (optional) validation data loader, and training data sampler.
    """
    train_dataset, valid_dataset = get_train_val_datasets(opts)

    n_train_samples = get_num_data_samples_as_int_or_mapping(train_dataset)

    is_master_node = is_master(opts)

    train_sampler = build_sampler(
        opts=opts,
        n_data_samples=n_train_samples,
        is_training=True,
        get_item_metadata=train_dataset.get_item_metadata,
    )
    if valid_dataset is not None:
        n_valid_samples = get_num_data_samples_as_int_or_mapping(valid_dataset)
        valid_sampler = build_sampler(
            opts=opts,
            n_data_samples=n_valid_samples,
            is_training=False,
            get_item_metadata=valid_dataset.get_item_metadata,
        )
    else:
        valid_sampler = None

    data_workers = getattr(opts, "dataset.workers", 1)
    persistent_workers = getattr(opts, "dataset.persistent_workers", False) and (
        data_workers > 0
    )
    pin_memory = getattr(opts, "dataset.pin_memory", False)
    prefetch_factor = getattr(opts, "dataset.prefetch_factor", 2)

    collate_fn_train, collate_fn_val = build_collate_fn(opts=opts)

    train_loader = CVNetsDataLoader(
        dataset=train_dataset,
        batch_size=1,  # Handled inside data sampler
        num_workers=data_workers,
        pin_memory=pin_memory,
        batch_sampler=train_sampler,
        persistent_workers=persistent_workers,
        collate_fn=partial(collate_fn_train, opts=opts)
        if collate_fn_train is not None
        else None,
        prefetch_factor=prefetch_factor,
    )

    if valid_dataset is not None:
        val_loader = CVNetsDataLoader(
            dataset=valid_dataset,
            batch_size=1,
            batch_sampler=valid_sampler,
            num_workers=data_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=partial(collate_fn_val, opts=opts)
            if collate_fn_val is not None
            else None,
        )
    else:
        val_loader = None

    if is_master_node:
        logger.log("Training sampler details: ")
        print("{}".format(train_sampler))

        if valid_dataset is not None:
            logger.log("Validation sampler details: ")
            print("{}".format(valid_sampler))
            logger.log("Number of data workers: {}".format(data_workers))

    return train_loader, val_loader, train_sampler


def get_num_data_samples_as_int_or_mapping(
    dataset: BaseDataset,
) -> Union[int, Mapping[str, int]]:
    """Return the number of samples in the dataset.

    The dataset can be a single or composition of multiple datasets (as in multi-task learning). For a single
    dataset, the number of samples is integer while for multiple datasets, a dictionary is returned with task name and
    number of samples per task.

    Args:
        dataset: An instance of `data.datasets.BaseDataset` class

    Returns:
        An integer for single dataset and mapping for composite datasets.

    """
    if hasattr(dataset, "get_dataset_length_as_mapping"):
        return dataset.get_dataset_length_as_mapping()
    else:
        return len(dataset)

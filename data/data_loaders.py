#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch.utils.data.dataloader import DataLoader
from utils import logger
from utils.ddp_utils import is_master

from .datasets import train_val_datasets, evaluation_datasets
from .sampler import build_sampler


def create_eval_loader(opts):
    eval_dataset = evaluation_datasets(opts)
    n_eval_samples = len(eval_dataset)
    is_master_node = is_master(opts)

    # overwrite the validation argument
    setattr(opts, "dataset.val_batch_size0", getattr(opts, "dataset.eval_batch_size0", 1))

    # we don't need variable batch sampler for evaluation
    sampler_name = getattr(opts, "sampler.name", "batch_sampler")
    if sampler_name.find("var") > -1:
        crop_size_w = getattr(opts, "sampler.vbs.crop_size_width", 224)
        crop_size_h = getattr(opts, "sampler.vbs.crop_size_height", 224)
        setattr(opts, "sampler.name", "batch_sampler")
        setattr(opts, "sampler.bs.crop_size_width", crop_size_w)
        setattr(opts, "sampler.bs.crop_size_height", crop_size_h)

    eval_sampler = build_sampler(opts=opts, n_data_samples=n_eval_samples, is_training=False)

    data_workers = getattr(opts, "dataset.workers", 1)
    persistent_workers = False
    pin_memory = False

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=1,
                                              batch_sampler=eval_sampler,
                                              num_workers=data_workers,
                                              pin_memory=pin_memory,
                                              persistent_workers=persistent_workers
                                              )

    if is_master_node:
        logger.log('Evaluation sampler details: ')
        print("{}".format(eval_sampler))

    return eval_loader


def create_train_val_loader(opts):
    train_dataset, valid_dataset = train_val_datasets(opts)

    n_train_samples = len(train_dataset)
    n_valid_samples = len(valid_dataset)
    is_master_node = is_master(opts)

    train_sampler = build_sampler(opts=opts, n_data_samples=n_train_samples, is_training=True)
    valid_sampler = build_sampler(opts=opts, n_data_samples=n_valid_samples, is_training=False)

    data_workers = getattr(opts, "dataset.workers", 1)
    persistent_workers = getattr(opts, "dataset.persistent_workers", False) and (data_workers > 0)
    pin_memory = getattr(opts, "dataset.pin_memory", False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,  # Handled inside data sampler
                              num_workers=data_workers,
                              pin_memory=pin_memory,
                              batch_sampler=train_sampler,
                              persistent_workers=persistent_workers
                              )

    val_loader = DataLoader(dataset=valid_dataset,
                            batch_size=1,
                            batch_sampler=valid_sampler,
                            num_workers=data_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers
                            )

    if is_master_node:
        logger.log('Training sampler details: ')
        print("{}".format(train_sampler))
        logger.log('Validation sampler details: ')
        print("{}".format(valid_sampler))
        logger.log("Number of data workers: {}".format(data_workers))

    return train_loader, val_loader, train_sampler

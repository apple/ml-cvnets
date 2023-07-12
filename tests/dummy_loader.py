#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from functools import partial

from data.collate_fns import build_collate_fn
from data.loader.dataloader import CVNetsDataLoader
from data.sampler import build_sampler
from tests.dummy_datasets import train_val_datasets
from utils import logger
from utils.ddp_utils import is_master


def create_train_val_loader(opts):
    train_dataset, valid_dataset = train_val_datasets(opts)

    n_train_samples = len(train_dataset)
    is_master_node = is_master(opts)

    train_sampler = build_sampler(
        opts=opts, n_data_samples=n_train_samples, is_training=True
    )
    if valid_dataset is not None:
        n_valid_samples = len(valid_dataset)
        valid_sampler = build_sampler(
            opts=opts, n_data_samples=n_valid_samples, is_training=False
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

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import List, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing import errors

from common import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_ITERATIONS,
)
from cvnets import EMA, get_model
from data import create_train_val_loader
from engine import Trainer
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from options.opts import get_training_arguments
from utils import logger, resources
from utils.checkpoint_utils import load_checkpoint, load_model_state
from utils.common_utils import create_directories, device_setup
from utils.ddp_utils import distributed_init, is_master


@errors.record
def main(opts: argparse.Namespace, **kwargs) -> None:
    # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))
    use_distributed = getattr(opts, "ddp.use_distributed")

    is_master_node = is_master(opts)

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # compute max iterations based on max epochs
    # Useful in doing polynomial decay
    is_iteration_based = getattr(opts, "scheduler.is_iteration_based")
    if is_iteration_based:
        max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        if max_iter is None or max_iter <= 0:
            logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
            setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
            max_iter = DEFAULT_ITERATIONS
        setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
        if is_master_node:
            logger.log("Max. iteration for training: {}".format(max_iter))
    else:
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if max_epochs is None or max_epochs <= 0:
            logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
            setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if is_master_node:
            logger.log("Max. epochs for training: {}".format(max_epochs))
    # set-up the model
    model = get_model(opts)
    # print model information on master node
    if is_master_node:
        model.info()

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last")
        else torch.contiguous_format
    )

    model = model.to(device=device, memory_format=memory_format)

    if getattr(opts, "ddp.use_deprecated_data_parallel"):
        logger.warning(
            "DataParallel is not recommended for training, and is not tested exhaustively. \
                Please use it only for debugging purposes. We will deprecated the support for DataParallel in future and \
                    encourage you to use DistributedDataParallel."
        )
        model = model.to(memory_format=memory_format, device=torch.device("cpu"))
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
    elif use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dev_id],
            output_device=dev_id,
            find_unused_parameters=getattr(opts, "ddp.find_unused_params"),
        )
        if is_master_node:
            logger.log("Using DistributedDataParallel for training")

    # create loss function, print its information, and move to device
    criteria = build_loss_fn(opts)
    if is_master_node:
        logger.log(logger.color_text("Loss function"))
        print(criteria)
    criteria = criteria.to(device=device)

    # create the optimizer and print its information
    optimizer = build_optimizer(model, opts=opts)
    if is_master_node:
        logger.log(logger.color_text("Optimizer"))
        print(optimizer)

    # create the gradient scalar
    gradient_scaler = GradScaler(enabled=getattr(opts, "common.mixed_precision"))

    # LR scheduler
    scheduler = build_scheduler(opts=opts)
    if is_master_node:
        logger.log(logger.color_text("Learning rate scheduler"))
        print(scheduler)

    model_ema = None
    use_ema = getattr(opts, "ema.enable")

    if use_ema:
        ema_momentum = getattr(opts, "ema.momentum")
        model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
        if is_master_node:
            logger.log("Using EMA")

    best_metric = 0.0 if getattr(opts, "stats.checkpoint_metric_max") else math.inf

    start_epoch = 0
    start_iteration = 0
    resume_loc = getattr(opts, "common.resume")
    finetune_loc = getattr(opts, "common.finetune")
    auto_resume = getattr(opts, "common.auto_resume")
    if resume_loc is not None or auto_resume:
        (
            model,
            optimizer,
            gradient_scaler,
            start_epoch,
            start_iteration,
            best_metric,
            model_ema,
        ) = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            model_ema=model_ema,
            gradient_scaler=gradient_scaler,
        )
    elif finetune_loc is not None:
        model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)
        if is_master_node:
            logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

    training_engine = Trainer(
        opts=opts,
        model=model,
        validation_loader=val_loader,
        training_loader=train_loader,
        optimizer=optimizer,
        criterion=criteria,
        scheduler=scheduler,
        start_epoch=start_epoch,
        start_iteration=start_iteration,
        best_metric=best_metric,
        model_ema=model_ema,
        gradient_scaler=gradient_scaler,
    )

    training_engine.run(train_sampler=train_sampler)


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get("start_rank", 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker(args: Optional[List[str]] = None, **kwargs):
    opts = get_training_arguments(args=args)
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank")
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc")
    run_label = getattr(opts, "common.run_label")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus")
    world_size = getattr(opts, "ddp.world_size")

    # use DDP if num_gpus is > 1
    use_distributed = True if num_gpus > 1 else False
    setattr(opts, "ddp.use_distributed", use_distributed)

    if num_gpus > 0:
        assert torch.cuda.is_available(), "We need CUDA for training on GPUs."

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = resources.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if getattr(opts, "ddp.use_deprecated_data_parallel") or num_gpus <= 1:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0") * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0") * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)

    else:
        # DDP is the default for training

        # get device id
        dev_id = getattr(opts, "ddp.device_id")
        # set the dev.device_id to the same as ddp.device_id.
        # note that dev arguments are not accessible through CLI.
        setattr(opts, "dev.device_id", dev_id)

        if world_size == -1:
            logger.log(
                "Setting --ddp.world-size the same as the number of available gpus"
            )
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // num_gpus)

        start_rank = getattr(opts, "ddp.rank")
        # we need to set rank to None as we set it inside the distributed_worker
        setattr(opts, "ddp.rank", None)
        kwargs["start_rank"] = start_rank
        setattr(opts, "ddp.start_rank", start_rank)
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )


if __name__ == "__main__":
    main_worker()

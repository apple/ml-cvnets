#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import multiprocessing
import torch

from utils import logger
from options.opts import get_loss_landscape_args
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from cvnets import get_model
from loss_fn import build_loss_fn
from data import create_eval_loader
from engine import Trainer


def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0)  # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    is_master_node = is_master(opts)

    # set-up data loaders
    val_loader = create_eval_loader(opts)

    # set-up the model
    model = get_model(opts)

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )

    if num_gpus == 0:
        logger.warning("Need atleast 1 GPU for training. Got {} GPUs.".format(num_gpus))
        model = model.to(device=device, memory_format=memory_format)
    elif num_gpus == 1:
        model = model.to(device=device, memory_format=memory_format)
    elif is_distributed:
        model = model.to(device=device, memory_format=memory_format)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dev_id],
            output_device=dev_id,
            find_unused_parameters=getattr(opts, "ddp.find_unused_params", False),
        )
        if is_master_node:
            logger.log("Using DistributedDataParallel for training")
    else:
        model = model.to(memory_format=memory_format)
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log("Using DataParallel for training")

    # setup criteria
    criteria = build_loss_fn(opts)
    criteria = criteria.to(device=device)

    training_engine = Trainer(
        opts=opts,
        model=model,
        validation_loader=val_loader,
        training_loader=None,
        optimizer=None,
        criterion=criteria,
        scheduler=None,
        start_epoch=0,
        start_iteration=0,
        best_metric=0,
        model_ema=None,
        gradient_scalar=None,
    )
    training_engine.run_loss_landscape()


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


def main_worker_loss_landscape(**kwargs):
    opts = get_loss_landscape_args()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    world_size = getattr(opts, "ddp.world_size", -1)
    use_distributed = not getattr(opts, "ddp.disable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")
    ddp_spawn = not getattr(opts, "ddp.no_spawn", False)
    if use_distributed and ddp_spawn and torch.cuda.is_available():
        # get device id
        dev_id = getattr(opts, "ddp.device_id", None)
        setattr(opts, "dev.device_id", dev_id)

        if world_size == -1:
            logger.log(
                "Setting --ddp.world-size the same as the number of available gpus"
            )
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // num_gpus)

        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs["start_rank"] = start_rank
        setattr(opts, "ddp.start_rank", start_rank)
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        if norm_name in ["sync_batch_norm", "sbn"]:
            setattr(opts, "model.normalization.name", "batch_norm")

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


if __name__ == "__main__":
    #
    main_worker_loss_landscape()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
import multiprocessing


from cvnets import get_model
from data import create_eval_loader
from engine import Evaluator

from options.opts import get_eval_arguments
from utils import logger
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init


def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0) # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device('cpu'))
    device = getattr(opts, "dev.device", torch.device('cpu'))
    is_distributed = getattr(opts, "ddp.use_distributed", False)

    # set-up data loaders
    val_loader = create_eval_loader(opts)

    # set-up the model
    model = get_model(opts)

    is_master_node = is_master(opts)
    if num_gpus <= 1:
        model = model.to(device=device)
    elif is_distributed:
        model = model.to(device=device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if is_master_node:
            logger.log('Using DistributedDataParallel for evaluation')
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log('Using DataParallel for evaluation')

    eval_engine = Evaluator(opts=opts, model=model, eval_loader=val_loader)
    eval_engine.run()


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    if torch.cuda.is_available():
        torch.cuda.set_device(i)

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get('start_rank', 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker(**kwargs):
    opts = get_eval_arguments()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    world_size = getattr(opts, "ddp.world_size", 1)
    num_gpus = getattr(opts, "dev.num_gpus", 1)
    use_distributed = getattr(opts, "ddp.enable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if use_distributed:
        if world_size == -1:
            logger.log("Setting --ddp.world-size the same as the number of available gpus")
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)
        elif world_size != num_gpus:
            logger.log("--ddp.world-size does not match num. of available GPUs. Got {} !={}".format(world_size, num_gpus))
            logger.log("Setting --ddp.world-size={}".format(num_gpus))
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // world_size)
            
        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs['start_rank'] = start_rank
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


# for segmentation and detection, we follow a different evaluation pipeline that allows to save the results too
def main_worker_segmentation(**kwargs):
    from engine.eval_segmentation import main_segmentation_evaluation
    main_segmentation_evaluation(**kwargs)


def main_worker_detection(**kwargs):
    from engine.eval_detection import main_detection_evaluation
    main_detection_evaluation(**kwargs)


if __name__ == "__main__":
    #main_worker()
    main_worker_segmentation()
    #main_worker_detection()

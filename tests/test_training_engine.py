#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os.path
import sys
from pathlib import Path
import pytest

sys.path.append("..")

import torch
import shutil
import multiprocessing
import math
from torch.cuda.amp import GradScaler

from cvnets import get_model, EMA
from loss_fn import build_loss_fn

from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master
from optim import build_optimizer
from optim.scheduler import build_scheduler
from utils.checkpoint_utils import load_checkpoint, load_model_state
from engine import Trainer

from tests.dummy_loader import create_train_val_loader
from tests.configs import get_config


def main(opts, is_iteration_based, **kwargs):
    device = getattr(opts, "dev.device", torch.device("cpu"))

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # compute max iterations based on max epochs
    # Useful in doing polynomial decay
    setattr(opts, "scheduler.is_iteration_based", is_iteration_based)
    if is_iteration_based:
        setattr(opts, "scheduler.max_iterations", 5)
        setattr(opts, "scheduler.max_epochs", 100000)
    else:
        setattr(opts, "scheduler.max_epochs", 2)
        setattr(opts, "scheduler.max_iterations", 10000)

    # set-up the model
    model = get_model(opts)

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )

    model = model.to(device=device, memory_format=memory_format)

    # setup criteria
    criteria = build_loss_fn(opts)
    criteria = criteria.to(device=device)

    # create the optimizer
    optimizer = build_optimizer(model, opts=opts)

    # create the gradient scalar
    gradient_scalar = GradScaler(enabled=False)

    # LR scheduler
    scheduler = build_scheduler(opts=opts)

    model_ema = None
    use_ema = getattr(opts, "ema.enable", False)

    if use_ema:
        ema_momentum = getattr(opts, "ema.momentum", 0.0001)
        model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)

    best_metric = (
        0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
    )

    start_epoch = 0
    start_iteration = 0
    resume_loc = getattr(opts, "common.resume", None)
    finetune_loc = getattr(opts, "common.finetune_imagenet1k", None)
    auto_resume = getattr(opts, "common.auto_resume", False)
    if resume_loc is not None or auto_resume:
        (
            model,
            optimizer,
            gradient_scalar,
            start_epoch,
            start_iteration,
            best_metric,
            model_ema,
        ) = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            model_ema=model_ema,
            gradient_scalar=gradient_scalar,
        )
    elif finetune_loc is not None:
        model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)

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
        gradient_scalar=gradient_scalar,
    )

    try:
        training_engine.run(train_sampler=train_sampler)
    except SystemExit as sys_exit:
        if str(sys_exit).find("Nan encountered in the loss") > -1:
            # Because we use random inputs and labels, loss can be NAN. If that is the case, skip the test.
            pytest.skip(str(sys_exit))


@pytest.mark.parametrize(
    ("config_file", "is_iteration_based"),
    [
        ("config/classification/imagenet/mobilevit.yaml", False),
        ("config/classification/imagenet/mobilevit_v2.yaml", True),
        ("config/detection/ssd_coco/resnet.yaml", False),
        ("config/segmentation/ade20k/deeplabv3_mobilenetv2.yaml", False),
        ("config/video_classification/kinetics/mobilevit_st.yaml", False),
        ("config/multi_modal_img_text/clip_vit.yaml", False),
    ],
)
def test_training_engine(config_file: str, is_iteration_based: bool):
    opts = get_config(config_file=config_file)

    # device set-up
    opts = device_setup(opts)

    setattr(opts, "ddp.rank", 0)

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)

    # if results directory exists, delete it
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    setattr(opts, "dev.num_gpus", 0)
    setattr(opts, "ddp.use_distributed", False)
    setattr(opts, "dataset.workers", 1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")

    if norm_name is not None and norm_name in ["sync_batch_norm", "sbn"]:
        setattr(opts, "model.normalization.name", "batch_norm")

    # adjust the batch size
    setattr(opts, "dataset.train_batch_size0", 2)
    setattr(opts, "dataset.val_batch_size0", 2)
    setattr(opts, "dev.device_id", None)
    setattr(opts, "dev.device", torch.device("cpu"))

    setattr(opts, "sampler.vbs.crop_size_width", 32)
    setattr(opts, "sampler.vbs.crop_size_height", 32)
    setattr(opts, "sampler.bs.crop_size_width", 32)
    setattr(opts, "sampler.bs.crop_size_height", 32)
    
    # We need to disable mixed_precision if testing on CPU only.
    setattr(opts, "common.mixed_precision", False)

    main(opts=opts, is_iteration_based=is_iteration_based)

    # delete the results directory folder
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

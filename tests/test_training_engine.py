#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os.path
import sys
from pathlib import Path

import pytest

sys.path.append("..")

import math
import random
import shutil

import torch
from torch.cuda.amp import GradScaler

from cvnets import EMA, get_model
from engine import Trainer
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from tests.configs import get_config
from tests.dummy_loader import create_train_val_loader
from tests.test_utils import unset_pretrained_models_from_opts
from utils.checkpoint_utils import load_checkpoint, load_model_state
from utils.common_utils import create_directories, device_setup
from utils.ddp_utils import is_master


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
    gradient_scaler = GradScaler(enabled=False)

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
    finetune_loc = getattr(opts, "common.finetune", None)
    auto_resume = getattr(opts, "common.auto_resume", False)
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


@pytest.mark.parametrize(
    ("config_file", "is_iteration_based"),
    [
        ("config/classification/imagenet/mobilevit.yaml", False),
        ("config/classification/imagenet/mobilevit_v2.yaml", True),
        ("config/detection/ssd_coco/resnet.yaml", False),
        ("config/segmentation/ade20k/deeplabv3_mobilenetv2.yaml", False),
        ("config/multi_modal_img_text/clip_vit.yaml", False),
        # add a configuration to test range augment
        ("examples/range_augment/classification/efficientnet_b0.yaml", False),
    ],
)
def test_training_engine(
    config_file: str, is_iteration_based: bool, tmp_path: Path
) -> None:
    opts = get_config(config_file=config_file)

    # device set-up
    opts = device_setup(opts)

    setattr(opts, "ddp.rank", 0)

    is_master_node = is_master(opts)

    # create the directory for saving results
    # Parallel tests causes issues when save_dir is accessed by multiple workers.
    # Therefore, we use a unique random path here and use that as a save location.
    save_dir = str(tmp_path)
    setattr(opts, "common.results_loc", save_dir)

    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)

    # if results directory exists, delete it
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    setattr(opts, "dev.num_gpus", 0)
    setattr(opts, "dataset.workers", 0)

    if getattr(opts, "dataset.name", "coco_ssd"):
        # coco_map metric requires access to instances_val2017.json file
        coco_annotations_path = os.path.join(os.path.dirname(__file__), "data", "coco")
        setattr(opts, "dataset.root_val", coco_annotations_path)

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
    setattr(opts, "sampler.vbs.min_crop_size_width", 32)
    setattr(opts, "sampler.vbs.min_crop_size_height", 32)
    setattr(opts, "sampler.vbs.max_crop_size_width", 64)
    setattr(opts, "sampler.vbs.max_crop_size_height", 64)

    # We need to disable mixed_precision if testing on CPU only.
    setattr(opts, "common.mixed_precision", False)

    # removing pretrained models (if any) for now to reduce test time as well as access issues
    unset_pretrained_models_from_opts(opts)

    main(opts=opts, is_iteration_based=is_iteration_based)

    # delete the results directory folder
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

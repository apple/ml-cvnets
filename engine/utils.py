#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from utils import logger
import torch
from torch import Tensor
from typing import Optional, Dict, Union, List, Any
import gc
from torch.cuda.amp import autocast

from utils.ddp_utils import is_master
from utils.tensor_utils import create_rand_tensor
from utils.common_utils import create_directories

str_to_torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def autocast_fn(enabled: bool, amp_precision: Optional[str] = "float16"):
    if enabled:
        # If AMP is enabled, ensure that:
        # 1. Device is CUDA
        # 2. dtype is FLOAT16 or BFLOAT16
        if amp_precision not in str_to_torch_dtype:
            logger.error(
                "For Mixed-precision training, supported dtypes are {}. Got: {}".format(
                    list(str_to_torch_dtype.keys()), amp_precision
                )
            )

        if not torch.cuda.is_available():
            logger.error("For mixed-precision training, CUDA device is required.")

        return autocast(enabled=enabled, dtype=str_to_torch_dtype[amp_precision])
    else:
        return autocast(enabled=False)


def print_summary(
    opts,
    model,
    criteria: Optional = None,
    optimizer: Optional = None,
    scheduler: Optional = None,
) -> None:
    if is_master(opts):
        logger.log(logger.color_text("Model"))
        print(model)
        dev = getattr(opts, "dev.device", torch.device("cpu"))
        try:
            inp_tensor = create_rand_tensor(opts, device=dev)

            if hasattr(model, "module"):
                model.module.profile_model(inp_tensor)
            else:
                model.profile_model(inp_tensor)
            del inp_tensor
        except Exception as e:
            pass

        if criteria is not None:
            # print criteria
            logger.log(logger.color_text("Loss function"))
            print("{}".format(criteria))

        if optimizer is not None:
            logger.log(logger.color_text("Optimizer"))
            print("{}".format(optimizer))

        if scheduler is not None:
            logger.log(logger.color_text("Learning rate scheduler"))
            print("{}".format(scheduler))

        gc.collect()


def get_batch_size(x: Union[Tensor, Dict, List]) -> int:
    if isinstance(x, Tensor):
        return x.shape[0]
    elif isinstance(x, Dict) and "image" in x:
        return get_batch_size(x["image"])
    elif isinstance(x, List):
        return len(x)
    else:
        raise NotImplementedError(f"Invalid type {type(x)}")


def log_metrics(
    lrs: Union[List, float],
    log_writer,
    train_loss: float,
    val_loss: float,
    epoch: int,
    best_metric: float,
    val_ema_loss: Optional[float] = None,
    ckpt_metric_name: Optional[str] = None,
    train_ckpt_metric: Optional[float] = None,
    val_ckpt_metric: Optional[float] = None,
    val_ema_ckpt_metric: Optional[float] = None,
) -> None:
    if not isinstance(lrs, list):
        lrs = [lrs]
    for g_id, lr_val in enumerate(lrs):
        log_writer.add_scalar("LR/Group-{}".format(g_id), round(lr_val, 6), epoch)

    log_writer.add_scalar("Train/Loss", round(train_loss, 2), epoch)
    log_writer.add_scalar("Val/Loss", round(val_loss, 2), epoch)
    log_writer.add_scalar("Common/Best Metric", round(best_metric, 2), epoch)
    if val_ema_loss is not None:
        log_writer.add_scalar("Val_EMA/Loss", round(val_ema_loss, 2), epoch)

    # If val checkpoint metric is different from loss, add that too
    if ckpt_metric_name is not None and ckpt_metric_name != "loss":
        if train_ckpt_metric is not None:
            log_writer.add_scalar(
                "Train/{}".format(ckpt_metric_name.title()),
                round(train_ckpt_metric, 2),
                epoch,
            )
        if val_ckpt_metric is not None:
            log_writer.add_scalar(
                "Val/{}".format(ckpt_metric_name.title()),
                round(val_ckpt_metric, 2),
                epoch,
            )
        if val_ema_ckpt_metric is not None:
            log_writer.add_scalar(
                "Val_EMA/{}".format(ckpt_metric_name.title()),
                round(val_ema_ckpt_metric, 2),
                epoch,
            )


def get_log_writers(opts: Dict[str, Any], save_location: Optional[str]):
    is_master_node = is_master(opts)

    log_writers = []
    if not is_master_node:
        return log_writers

    tensorboard_logging = getattr(opts, "common.tensorboard_logging", False)
    if tensorboard_logging and save_location is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            logger.log(
                "Unable to import SummaryWriter from torch.utils.tensorboard. Disabling tensorboard logging"
            )
            SummaryWriter = None

        if SummaryWriter is not None:
            exp_dir = "{}/tb_logs".format(save_location)
            create_directories(dir_path=exp_dir, is_master_node=is_master_node)
            log_writers.append(
                SummaryWriter(log_dir=exp_dir, comment="Training and Validation logs")
            )

    bolt_logging = getattr(opts, "common.bolt_logging", False)
    if bolt_logging:
        try:
            from internal.utils.bolt_logger import BoltLogger
        except ModuleNotFoundError:
            BoltLogger = None

        if BoltLogger is None:
            logger.log("Unable to import bolt. Disabling bolt logging")
        else:
            log_writers.append(BoltLogger())

    return log_writers

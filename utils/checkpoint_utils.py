#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch

from cvnets import EMA
from optim import BaseOptim
from utils import logger
from utils.common_utils import unwrap_model_fn
from utils.ddp_utils import is_master
from utils.download_utils import get_local_path

CHECKPOINT_EXTN = "pt"


def get_model_state_dict(model: torch.nn.Module) -> Dict:
    """Returns `state_dict` of a given model.

    Args:
        model: A torch model (it can be also a wrapped model, e.g., with DDP).

    Returns:
        `state_dict` of the model. If model is an EMA instance, the `state_dict` corresponding to EMA parameters is
            returned.
    """
    if isinstance(model, EMA):
        return get_model_state_dict(model.ema_model)
    else:
        unwrapped_model = unwrap_model_fn(model)
        return unwrapped_model.state_dict()


def load_state_dict(
    model: torch.nn.Module, state_dict: Dict, strict: bool = True
) -> torch.nn.Module:
    """Load the given `state_dict` into the model.

    Args:
        model: A torch model (it can be also a wrapped model, e.g., with DDP).
        state_dict: A state dict dictionary to load model parameters from.
        strict: whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
            `state_dict` function. Default: ``True``.

    Returns:
        model loaded with parameters from the given state_dict
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)
    return model


def average_ckpts(ckpt_loc_list: List[str]) -> Dict:
    """Compute averaged parameters from a list of checkpoints.

    Args:
        ckpt_loc_list: List of paths to model checkpoints to be averaged.

    Returns:
        `state_dict` corresponding to the averaged parameters.
    """
    avg_state_dict = dict()
    key_count = dict()
    key_dtype = dict()

    for c in ckpt_loc_list:
        if not os.path.isfile(c):
            pass
        ckpt_state_dict = torch.load(c, map_location="cpu")

        for k, v in ckpt_state_dict.items():
            if k not in avg_state_dict:
                key_dtype[k] = v.dtype
                avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                key_count[k] = 1.0
            else:
                avg_state_dict[k] += v.to(dtype=torch.float64)
                key_count[k] += 1.0

    for k, v in avg_state_dict.items():
        avg_state_dict[k] = v.div(key_count[k]).to(dtype=key_dtype[k])
    return avg_state_dict


def avg_and_save_k_checkpoints(
    model_state: Dict,
    best_metric: float,
    k_best_checkpoints: int,
    max_ckpt_metric: bool,
    ckpt_str: str,
) -> None:
    """Save top-k checkpoints and their average.

    Args:
        model_state: `state_dict` containing model parameters.
        best_metric: Best observed value of the tracking validation metric. For example, best top-1 validation accuracy
            that is observed until the current iteration.
        k_best_checkpoints: An integer k determining number of top (based on validation metric) checkpoints to keep.
            If `k_best_checkpoints` is smaller than 1, only best checkpoint is stored.
        max_ckpt_metric: A boolean demonstrating whether the tracking validation metric is higher the better, or lower
            the better.
        ckpt_str: String determining path prefix for checkpoints to be saved.
    """
    try:
        ckpt_fname = "{}_score_{:.4f}.{}".format(ckpt_str, best_metric, CHECKPOINT_EXTN)
        torch.save(model_state, ckpt_fname)

        best_fnames = glob.glob("{}_score_*".format(ckpt_str))
        best_scores = [
            float(f.split("_score_")[-1].replace(".{}".format(CHECKPOINT_EXTN), ""))
            for f in best_fnames
        ]

        best_scores_keep = []
        if len(best_scores) > k_best_checkpoints:
            best_scores = sorted(best_scores)
            if not max_ckpt_metric:
                best_scores = best_scores[::-1]
            best_scores_keep = best_scores[-k_best_checkpoints:]
            for k in best_scores:
                if k in best_scores_keep:
                    continue
                rm_ckpt = "{}_score_{:.4f}.{}".format(ckpt_str, k, CHECKPOINT_EXTN)
                os.remove(rm_ckpt)
                logger.log("Deleting checkpoint: {}".format(rm_ckpt))
        #
        if len(best_scores_keep) > 1:
            avg_fnames = [
                "{}_score_{:.4f}.{}".format(ckpt_str, k, CHECKPOINT_EXTN)
                for k in best_scores_keep
            ]
            logger.log(
                "Averaging checkpoints: {}".format(
                    [f.split("/")[-1] for f in avg_fnames]
                )
            )
            # save the average model
            avg_model_state = average_ckpts(ckpt_loc_list=avg_fnames)
            ckpt_fname = "{}_avg.{}".format(ckpt_str, CHECKPOINT_EXTN)
            if avg_model_state:
                torch.save(avg_model_state, ckpt_fname)
                logger.log("Averaged checkpoint saved at: {}".format(ckpt_fname))
    except Exception as e:
        logger.log("Error in k-best-checkpoint")
        print(e)


def save_interval_checkpoint(
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Union[BaseOptim, torch.optim.Optimizer],
    best_metric: float,
    save_dir: str,
    gradient_scaler: torch.cuda.amp.GradScaler,
    model_ema: Optional[torch.nn.Module] = None,
    *args,
    **kwargs
) -> None:
    """Save current iteration training checkpoint.

    Args:
        iterations: An integer denoting training iteration number. Each iteration corresponds to forward-backward passes
        on a batch with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store training optimization state variables.
        best_metric: Best observed value of the tracking validation metric. For example, best top-1 validation accuracy
            that is observed until the current iteration.
        save_dir: Path to a directory to save checkpoints.
        gradient_scaler: `GradScaler` object storing required automatic mixed precision state.
        model_ema: EMA model to be stored in the checkpoint.
    """
    checkpoint = get_training_state(
        iterations, epoch, model, optimizer, best_metric, gradient_scaler, model_ema
    )
    ckpt_str = "{}/training_checkpoint".format(save_dir)
    ckpt_fname = "{}_{}_{}.{}".format(ckpt_str, epoch, iterations, CHECKPOINT_EXTN)
    torch.save(checkpoint, ckpt_fname)


def get_training_state(
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Union[BaseOptim, torch.optim.Optimizer],
    best_metric: float,
    gradient_scaler: torch.cuda.amp.GradScaler,
    model_ema: Optional[torch.nn.Module] = None,
) -> Dict:
    """Create a checkpoint dictionary that includes all required states to resume the training from its current state.

    Args:
        iterations: An integer denoting training iteration number. Each iteration corresponds to forward-backward passes
            on a batch with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store training optimization state variables.
        best_metric: Best observed value of the tracking validation metric. For example, best top-1 validation accuracy
            that is observed until the current iteration.
        gradient_scaler: `GradScaler` object storing required automatic mixed precision state.
        model_ema: EMA model to be stored in the checkpoint.

    Returns:
        A dictionary that includes all required states to resume the training from its current state.
    """
    model_state = get_model_state_dict(model)
    training_state = {
        "iterations": iterations,
        "epoch": epoch,
        "model_state_dict": model_state,
        "optim_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "gradient_scaler_state_dict": gradient_scaler.state_dict(),
    }
    if model_ema is not None:
        training_state["ema_state_dict"] = get_model_state_dict(model_ema)
    return training_state


def save_checkpoint(
    iterations: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Union[BaseOptim, torch.optim.Optimizer],
    best_metric: float,
    is_best: bool,
    save_dir: str,
    gradient_scaler: torch.cuda.amp.GradScaler,
    model_ema: Optional[torch.nn.Module] = None,
    is_ema_best: bool = False,
    ema_best_metric: Optional[float] = None,
    max_ckpt_metric: bool = False,
    k_best_checkpoints: int = -1,
    save_all_checkpoints: bool = False,
    *args,
    **kwargs
) -> None:
    """Save checkpoints corresponding to the current state of the training.

    Args:
        iterations: An integer denoting training iteration number. Each iteration corresponds to forward-backward passes
            on a batch with all GPUs.
        epoch: An integer denoting epoch number.
        model: The model being trained.
        optimizer: Optimizer object, which possibly store training optimization state variables.
        best_metric: Best observed value of the tracking validation metric. For example, best top-1 validation accuracy
            that is observed until the current iteration.
        is_best: A boolean demonstrating whether the current model obtains the best validation metric compared
            to the previously saved checkpoints.
        save_dir: Path to a directory to save checkpoints.
        gradient_scaler: `GradScaler` object storing required automatic mixed precision state.
        model_ema: EMA model to be stored in the checkpoint.
        is_ema_best: A boolean demonstrating whether the current EMA model obtains the best validation metric compared
            to the previously saved checkpoints.
        ema_best_metric: Best observed value of the tracking validation metric by the EMA model.
        max_ckpt_metric: A boolean demonstrating whether the tracking validation metric is higher the better, or lower
            the better.
        k_best_checkpoints: An integer k determining number of top (based on validation metric) checkpoints to keep.
            If `k_best_checkpoints` is smaller than 1, only best checkpoint is stored.
        save_all_checkpoints: If True, will save model_state checkpoints (main model and its EMA) for all epochs.
    """
    checkpoint = get_training_state(
        iterations, epoch, model, optimizer, best_metric, gradient_scaler, model_ema
    )
    model_state = checkpoint.get("model_state_dict")
    ckpt_str = "{}/checkpoint".format(save_dir)

    if is_best:
        best_model_fname = "{}_best.{}".format(ckpt_str, CHECKPOINT_EXTN)
        if os.path.isfile(best_model_fname):
            os.remove(best_model_fname)
        torch.save(model_state, best_model_fname)
        logger.log(
            "Best checkpoint with score {:.2f} saved at {}".format(
                best_metric, best_model_fname
            )
        )

        if k_best_checkpoints > 1:
            avg_and_save_k_checkpoints(
                model_state, best_metric, k_best_checkpoints, max_ckpt_metric, ckpt_str
            )

    if model_ema is not None:
        ema_fname = "{}_ema_last.{}".format(ckpt_str, CHECKPOINT_EXTN)
        torch.save(checkpoint["ema_state_dict"], ema_fname)
        if save_all_checkpoints:
            ema_fname = "{}_ema_epoch{}.{}".format(ckpt_str, epoch, CHECKPOINT_EXTN)
            torch.save(checkpoint["ema_state_dict"], ema_fname)

        if is_ema_best:
            ema_best_fname = "{}_ema_best.{}".format(ckpt_str, CHECKPOINT_EXTN)
            if os.path.isfile(ema_best_fname):
                os.remove(ema_best_fname)
            torch.save(checkpoint["ema_state_dict"], ema_best_fname)
            logger.log(
                "Best EMA checkpoint with score {:.2f} saved at {}".format(
                    ema_best_metric, ema_best_fname
                )
            )

            if k_best_checkpoints > 1 and ema_best_metric is not None:
                avg_and_save_k_checkpoints(
                    model_state=checkpoint["ema_state_dict"],
                    best_metric=ema_best_metric,
                    k_best_checkpoints=k_best_checkpoints,
                    max_ckpt_metric=max_ckpt_metric,
                    ckpt_str="{}_ema".format(ckpt_str),
                )

    ckpt_fname = "{}/training_checkpoint_last.{}".format(save_dir, CHECKPOINT_EXTN)
    torch.save(checkpoint, ckpt_fname)

    ckpt_fname = "{}_last.{}".format(ckpt_str, CHECKPOINT_EXTN)
    torch.save(model_state, ckpt_fname)

    if save_all_checkpoints:
        ckpt_fname = "{}_epoch{}.{}".format(ckpt_str, epoch, CHECKPOINT_EXTN)
        torch.save(model_state, ckpt_fname)


def load_checkpoint(
    opts: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: Union[BaseOptim, torch.optim.Optimizer],
    gradient_scaler: torch.cuda.amp.GradScaler,
    model_ema: Optional[torch.nn.Module] = None,
) -> Tuple[
    torch.nn.Module,
    Union[BaseOptim, torch.optim.Optimizer],
    torch.cuda.amp.GradScaler,
    int,
    int,
    float,
    Optional[torch.nn.Module],
]:
    """Load a training checkpoint to resume training.

    Args:
        opts: Input arguments.
        model: The model to be loaded with `model_state_dict` from the checkpoint.
        optimizer: Optimizer object to be loaded with `optim_state_dict` from the checkpoint.
        gradient_scaler: A `GradScaler` object to be loaded with `gradient_scaler_state_dict`  from the checkpoint.
        model_ema: (Optional) EMA model to be loaded with `ema_state_dict` from the checkpoint.

    Returns:
        Tuple of loaded objects and value:
        (model, optimizer, gradient_scaler, start_epoch, start_iteration, best_metric, model_ema)
    """
    resume_loc = getattr(opts, "common.resume", None)
    dev_id = getattr(opts, "dev.device_id", None)
    device = getattr(opts, "dev.device", torch.device("cpu"))
    start_epoch = start_iteration = 0
    best_metric = (
        0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
    )
    auto_resume = getattr(opts, "common.auto_resume", False)
    exp_dir = getattr(opts, "common.exp_loc", None)
    is_master_node = is_master(opts)
    if resume_loc is None and auto_resume and exp_dir is not None:
        resume_loc = "{}/training_checkpoint_last.{}".format(exp_dir, CHECKPOINT_EXTN)

    resume_loc = get_local_path(opts, path=resume_loc)
    if resume_loc is not None and os.path.isfile(resume_loc):
        if dev_id is None:
            checkpoint = torch.load(resume_loc, map_location=device)
        else:
            checkpoint = torch.load(resume_loc, map_location="cuda:{}".format(dev_id))

        start_epoch = checkpoint["epoch"] + 1
        start_iteration = checkpoint["iterations"] + 1
        best_metric = checkpoint["best_metric"]

        model = load_state_dict(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state_dict"])

        if model_ema is not None and "ema_state_dict" in checkpoint:
            model_ema.ema_model = load_state_dict(
                model_ema.ema_model, checkpoint["ema_state_dict"]
            )

        if is_master_node:
            logger.log("Loaded checkpoint from {}".format(resume_loc))
            logger.log("Resuming training for epoch {}".format(start_epoch))
    else:
        if is_master_node:
            logger.log("No checkpoint found at '{}'".format(resume_loc))
    return (
        model,
        optimizer,
        gradient_scaler,
        start_epoch,
        start_iteration,
        best_metric,
        model_ema,
    )


def load_model_state(
    opts: argparse.Namespace,
    model: torch.nn.Module,
    model_ema: Optional[torch.nn.Module] = None,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    """Load the model (and optionally the EMA model) for finetuning.

    Args:
        opts: Input arguments.
        model: The model to be loaded with checkpoint at `common.finetune`.
        model_ema: The EMA model to be loaded with checkpoint at `common.finetune_ema`.

    Returns:
        Tuple of loaded model and EMA model. The second returned value is None when model_ema is not passed.
    """
    dev_id = getattr(opts, "dev.device_id", None)
    device = getattr(opts, "dev.device", torch.device("cpu"))
    finetune_loc = getattr(opts, "common.finetune", None)
    finetune_ema_loc = getattr(opts, "common.finetune_ema", None)

    def load_state(path):
        path = get_local_path(opts, path=path)
        if dev_id is None:
            model_state = torch.load(path, map_location=device)
        else:
            model_state = torch.load(path, map_location="cuda:{}".format(dev_id))
        return model_state

    if finetune_loc is not None and os.path.isfile(finetune_loc):
        # load model dict
        model = load_state_dict(model, load_state(finetune_loc))

        # load ema dict
        if model_ema is not None and os.path.isfile(finetune_ema_loc):
            model_ema = load_state_dict(model, load_state(finetune_ema_loc))

    return model, model_ema


def copy_weights(
    model_src: torch.nn.Module, model_tgt: torch.nn.Module
) -> torch.nn.Module:
    """Copy `state_dict` from source model to target model.

    Args:
        model_src: The source model.
        model_tgt: The target model.

    Returns:
        Target model with `state_dict` loaded from model_src.
    """
    with torch.no_grad():
        model_state = get_model_state_dict(model=model_src)
        return load_state_dict(model=model_tgt, state_dict=model_state)

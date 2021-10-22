#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import torch
from typing import Optional, Union
import math
import glob

from cvnets import EMA
from optim import BaseOptim
from utils import logger
from utils.ddp_utils import is_master
from utils.download_utils import get_local_path

CHECKPOINT_EXTN = "pt"


def get_model_state_dict(model):
    if isinstance(model, EMA):
        return get_model_state_dict(model.ema_model)
    else:
        return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def load_state_dict(model, state_dict):
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def average_ckpts(ckpt_loc_list: list):
    avg_state_dict = dict()
    key_count = dict()
    key_dtype = dict()

    for c in ckpt_loc_list:
        if not os.path.isfile(c):
            pass
        ckpt_state_dict = torch.load(c, map_location='cpu')

        for k, v in ckpt_state_dict.items():
            if k not in avg_state_dict:
                key_dtype[k] = v.dtype
                avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                key_count[k] = 1
            else:
                avg_state_dict[k] += v.to(dtype=torch.float64)
                key_count[k] += 1

    for k, v in avg_state_dict.items():
        avg_state_dict[k] = v.div(key_count[k]).to(dtype=key_dtype[k])
    return avg_state_dict


def avg_n_save_k_checkpoints(model_state, best_metric, k_best_checkpoints, max_ckpt_metric, ckpt_str):
    try:
        ckpt_fname = '{}_score_{:.4f}.{}'.format(ckpt_str, best_metric, CHECKPOINT_EXTN)
        torch.save(model_state, ckpt_fname)

        best_fnames = glob.glob('{}_score_*'.format(ckpt_str))
        best_scores = [float(f.split('_score_')[-1].replace(".{}".format(CHECKPOINT_EXTN), "")) for f in best_fnames]

        best_scores_keep = []
        if len(best_scores) > k_best_checkpoints:
            best_scores = sorted(best_scores)
            if not max_ckpt_metric:
                best_scores = best_scores[::-1]
            best_scores_keep = best_scores[-k_best_checkpoints:]
            for k in best_scores:
                if k in best_scores_keep:
                    continue
                rm_ckpt = '{}_score_{:.4f}.{}'.format(ckpt_str, k, CHECKPOINT_EXTN)
                os.remove(rm_ckpt)
                logger.log("Deleting checkpoint: {}".format(rm_ckpt))
        #
        if len(best_scores_keep) > 1:
            avg_fnames = ['{}_score_{:.4f}.{}'.format(ckpt_str, k, CHECKPOINT_EXTN) for k in best_scores_keep]
            logger.log("Averaging checkpoints: {}".format([f.split('/')[-1] for f in avg_fnames]))
            # save the average model
            avg_model_state = average_ckpts(ckpt_loc_list=avg_fnames)
            ckpt_fname = '{}_avg.{}'.format(ckpt_str, CHECKPOINT_EXTN)
            if avg_model_state:
                torch.save(avg_model_state, ckpt_fname)
                logger.log('Averaged checkpoint saved at: {}'.format(ckpt_fname))
    except Exception as e:
        logger.log("Error in k-best-checkpoint")
        print(e)


def save_checkpoint(iterations: int,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: Union[BaseOptim, torch.optim.Optimizer],
                    best_metric: float,
                    is_best: bool,
                    save_dir: str,
                    gradient_scalar: torch.cuda.amp.GradScaler,
                    model_ema: Optional[torch.nn.Module] = None,
                    is_ema_best: Optional[bool] = False,
                    ema_best_metric: Optional[float] = None,
                    max_ckpt_metric: Optional[bool] = False,
                    k_best_checkpoints: Optional[int] = -1,
                    *args, **kwargs) -> None:
    model_state = get_model_state_dict(model)
    checkpoint = {
        'iterations': iterations,
        'epoch': epoch,
        'model_state_dict': model_state,
        'optim_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'gradient_scalar_state_dict': gradient_scalar.state_dict()
    }
    ckpt_str = '{}/checkpoint'.format(save_dir)

    if is_best:
        best_model_fname = '{}_best.{}'.format(ckpt_str, CHECKPOINT_EXTN)
        torch.save(model_state, best_model_fname)

    if model_ema is not None:
        checkpoint['ema_state_dict'] = get_model_state_dict(model_ema)
        ema_fname = '{}_ema.{}'.format(ckpt_str, CHECKPOINT_EXTN)
        torch.save(checkpoint['ema_state_dict'], ema_fname)
        if is_ema_best:
            ema_best_fname = '{}_ema_best.{}'.format(ckpt_str, CHECKPOINT_EXTN)
            torch.save(checkpoint['ema_state_dict'], ema_best_fname)

    ckpt_fname = '{}.{}'.format(ckpt_str, CHECKPOINT_EXTN)
    torch.save(checkpoint, ckpt_fname)

    ckpt_fname = '{}_last.{}'.format(ckpt_str, CHECKPOINT_EXTN)
    torch.save(model_state, ckpt_fname)

    if k_best_checkpoints > 1:
        avg_n_save_k_checkpoints(model_state, best_metric, k_best_checkpoints, max_ckpt_metric, ckpt_str)
        if model_ema is not None and ema_best_metric is not None:
            avg_n_save_k_checkpoints(model_state=checkpoint['ema_state_dict'],
                                     best_metric=ema_best_metric,
                                     k_best_checkpoints=k_best_checkpoints,
                                     max_ckpt_metric=max_ckpt_metric,
                                     ckpt_str="{}_ema".format(ckpt_str)
                                     )


def load_checkpoint(opts,
                    model: torch.nn.Module,
                    optimizer: Union[BaseOptim, torch.optim.Optimizer],
                    gradient_scalar: torch.cuda.amp.GradScaler,
                    model_ema: Optional[torch.nn.Module] = None):
    resume_loc = getattr(opts, "common.resume", None)
    dev_id = getattr(opts, "dev.device_id", None)
    device = getattr(opts, "dev.device", torch.device('cpu'))
    start_epoch = start_iteration = 0
    best_metric = 0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
    auto_resume = getattr(opts, "common.auto_resume", False)
    exp_dir = getattr(opts, "common.exp_loc", None)
    is_master_node = is_master(opts)
    if resume_loc is None and auto_resume and exp_dir is not None:
        resume_loc = '{}/checkpoint.{}'.format(exp_dir, CHECKPOINT_EXTN)

    resume_loc = get_local_path(opts, path=resume_loc)
    if resume_loc is not None and os.path.isfile(resume_loc):
        if dev_id is None:
            checkpoint = torch.load(resume_loc, map_location=device)
        else:
            checkpoint = torch.load(resume_loc, map_location='cuda:{}'.format(dev_id))

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iterations'] + 1
        best_metric = checkpoint['best_metric']

        model = load_state_dict(model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        gradient_scalar.load_state_dict(checkpoint['gradient_scalar_state_dict'])

        if model_ema is not None and 'ema_state_dict' in checkpoint:
            model_ema.ema_model = load_state_dict(model_ema.ema_model, checkpoint['ema_state_dict'])

        if is_master_node:
            logger.log('Loaded checkpoint from {}'.format(resume_loc))
            logger.log('Resuming training for epoch {}'.format(start_epoch))
    else:
        if is_master_node:
            logger.log("No checkpoint found at '{}'".format(resume_loc))
    return model, optimizer, gradient_scalar, start_epoch, start_iteration, best_metric, model_ema


def load_model_state(opts, model, model_ema=None):
    dev_id = getattr(opts, "dev.device_id", None)
    device = getattr(opts, "dev.device", torch.device('cpu'))
    finetune_loc = getattr(opts, "common.finetune", None)
    finetune_ema_loc = getattr(opts, "common.finetune_ema", None)

    def load_state(path):
        path = get_local_path(opts, path=path)
        if dev_id is None:
            model_state = torch.load(path, map_location=device)
        else:
            model_state = torch.load(path, map_location='cuda:{}'.format(dev_id))
        return model_state

    if finetune_loc is not None and os.path.isfile(finetune_loc):
        # load model dict
        model = load_state_dict(model, load_state(finetune_loc))

        # load ema dict
        if model_ema is not None and os.path.isfile(finetune_ema_loc):
            model_ema = load_state_dict(model, load_state(finetune_ema_loc))

    return model, model_ema


def copy_weights(model_src: torch.nn.Module, model_tgt: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        model_state = get_model_state_dict(model=model_src)
        return load_state_dict(model=model_tgt, state_dict=model_state)



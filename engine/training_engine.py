#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys
import traceback

import torch
import copy
import gc
import time
import shutil
from typing import Dict
from torch.cuda.amp import autocast
from torch.nn import functional as F
import random
from typing import Union, List, Optional
import numpy as np
from itertools import product

from data.transforms.image_torch import RandomMixup, RandomCutmix
from engine.utils import print_summary
from metrics import Statistics, metric_monitor
from common import DEFAULT_ITERATIONS, DEFAULT_EPOCHS, DEFAULT_LOG_FREQ
from utils import logger
from utils.common_utils import create_directories, move_to_device
from utils.ddp_utils import is_master, dist_barrier
from utils.tensor_utils import reduce_tensor_sum, tensor_to_python_float
from utils.checkpoint_utils import copy_weights, save_checkpoint
from loss_landscape import landscape_utils as ll_utils


class Trainer(object):
    """
    This class defines the training and validation code for training models with CVNets
    """

    def __init__(
        self,
        opts,
        model,
        validation_loader,
        training_loader,
        criterion,
        optimizer,
        scheduler,
        gradient_scalar,
        start_epoch: int = 0,
        start_iteration: int = 0,
        best_metric: float = 0.0,
        model_ema=None,
        *args,
        **kwargs
    ) -> None:
        super(Trainer, self).__init__()

        self.opts = opts

        self.model = model
        self.model_ema = model_ema
        self.criteria = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_scalar = gradient_scalar

        self.val_loader = validation_loader
        self.train_loader = training_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))

        self.start_epoch = start_epoch
        self.best_metric = best_metric
        self.train_iterations = start_iteration

        self.is_master_node = is_master(opts)
        self.max_iterations_reached = False
        self.max_iterations = getattr(
            self.opts, "scheduler.max_iterations", DEFAULT_ITERATIONS
        )
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)
        self.accum_freq = getattr(self.opts, "common.accum_freq", 1)
        self.accum_after_epoch = getattr(self.opts, "common.accum_after_epoch", 0)

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)

        self.train_metric_names = getattr(opts, "stats.train", ["loss"])
        if isinstance(self.train_metric_names, str):
            self.train_metric_names = [self.train_metric_names]

        assert isinstance(
            self.train_metric_names, list
        ), "Type of metric names should be list. Got: {}".format(
            type(self.train_metric_names)
        )

        if "loss" not in self.train_metric_names:
            self.train_metric_names.append(self.train_metric_names)

        self.val_metric_names = getattr(opts, "stats.val", ["loss"])
        if isinstance(self.val_metric_names, str):
            self.val_metric_names = [self.val_metric_names]

        assert isinstance(
            self.val_metric_names, list
        ), "Type of metric names should be list. Got: {}".format(
            type(self.val_metric_names)
        )

        if "loss" not in self.val_metric_names:
            self.val_metric_names.append(self.val_metric_names)

        self.save_all_checkpoints = getattr(
            self.opts, "stats.save_all_checkpoints", False
        )
        self.ckpt_metric = getattr(self.opts, "stats.checkpoint_metric", "loss")
        if self.ckpt_metric is None:
            # if checkpoint metric is not specified, then use loss
            self.ckpt_metric = "loss"

        assert (
            self.ckpt_metric in self.val_metric_names
        ), "Checkpoint metric should be part of metric names. Metric names: {}, Checkpoint metric: {}".format(
            self.val_metric_names, self.ckpt_metric
        )
        self.ckpt_metric = self.ckpt_metric.lower()

        self.tb_log_writer = None
        self.bolt_log_writer = None
        if self.is_master_node:
            self.setup_log_writer()

            print_summary(
                opts=self.opts,
                model=self.model,
                criteria=self.criteria,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

        self.adjust_norm_mom = None
        if getattr(opts, "model.normalization.adjust_bn_momentum.enable", False):
            from cvnets.layers import AdjustBatchNormMomentum

            self.adjust_norm_mom = AdjustBatchNormMomentum(opts=opts)
            if self.is_master_node:
                logger.log(
                    "Batch normalization momentum will be annealed during training."
                )
                print(self.adjust_norm_mom)

        # sample-efficient training
        self.cache_dict = None
        self.sample_efficient_training = getattr(
            opts, "dataset.sample_efficient_training.enable", False
        )
        self.sample_confidence = getattr(
            opts, "dataset.sample_efficient_training.sample_confidence", 0.5
        )
        self.find_easy_samples_every_k_epoch = getattr(
            opts,
            "dataset.sample_efficient_training.find_easy_samples_every_k_epochs",
            5,
        )
        self.min_sample_frequency = getattr(
            opts, "dataset.sample_efficient_training.min_sample_frequency", 5
        )
        if self.sample_efficient_training:
            self.train_loader_set = copy.deepcopy(self.train_loader)
            self.sample_ids_orig = self.train_loader_set.get_sample_indices()
            n_samples = len(self.sample_ids_orig)
            self.running_sum_tensor = torch.zeros(
                (n_samples,), device=self.device, dtype=torch.int
            )
            self.running_sum_tensor.requires_grad = False
            if self.is_master_node:
                logger.log("Configuring for sample efficient training")

        # recent versions of PyTorch support setting grads to None, for better performance
        # To be explored in Future
        # self.optimizer.zero_grad(set_to_none=True)
        self.set_grad_to_none = False

    def setup_log_writer(self):
        tensorboard_logging = getattr(self.opts, "common.tensorboard_logging", False)
        if tensorboard_logging:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                logger.log(
                    "Unable to import SummaryWriter from torch.utils.tensorboard. Disabling tensorboard logging"
                )
                SummaryWriter = None

            if SummaryWriter is not None:
                exp_dir = getattr(self.opts, "common.exp_loc", "results/run_1")
                exp_dir = "{}/tb_logs".format(exp_dir)
                create_directories(dir_path=exp_dir, is_master_node=self.is_master_node)
                self.tb_log_writer = SummaryWriter(
                    log_dir=exp_dir, comment="Training and Validation logs"
                )
            else:
                self.tb_log_writer = None

        bolt_logging = getattr(self.opts, "common.bolt_logging", False)
        if bolt_logging:
            try:
                from utils.bolt_logger import BoltLogger
            except ModuleNotFoundError:
                BoltLogger = None

            if BoltLogger is None:
                logger.log("Unable to import bolt. Disabling bolt logging")
                self.bolt_log_writer = None
            else:
                self.bolt_log_writer = BoltLogger()

    def compute_grad_norm(self):
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None

        norm_type = 2.0  # L2 norm

        inv_scale = 1.0 / self.gradient_scalar.get_scale()
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach() * inv_scale, norm_type).to(self.device)
                    for p in parameters
                ]
            ),
            norm_type,
        )
        if total_norm.isnan() or total_norm.isinf():
            return None
        return total_norm

    def _get_batch_size(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        elif isinstance(x, Dict):
            return x["image"].shape[0]

    def apply_mixup_transforms(self, data):
        # Apply mixup transforms on classification tasks
        opts = self.opts
        mixup_transforms = []
        if getattr(opts, "image_augmentation.mixup.enable", False):
            n_classes = getattr(opts, "model.classification.n_classes", None)
            if n_classes is None:
                logger.error("Please specify number of classes. Got None.")
            mixup_transforms.append(RandomMixup(opts=opts, num_classes=n_classes))

        if getattr(opts, "image_augmentation.cutmix.enable", False):
            n_classes = getattr(opts, "model.classification.n_classes", None)
            if n_classes is None:
                logger.error("Please specify number of classes. Got None.")
            mixup_transforms.append(RandomCutmix(opts=opts, num_classes=n_classes))

        if len(mixup_transforms) > 0:
            _mixup_transform = random.choice(mixup_transforms)
            data = _mixup_transform(data)
        return data

    def _zero_grad(self):
        if self.set_grad_to_none:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad()

    def train_epoch(self, epoch):
        time.sleep(2)  # To prevent possible deadlock during epoch transition

        if self.is_master_node:
            logger.double_dash_line()
            logger.debug(
                "Training epoch {} with {} samples".format(
                    epoch, self.train_loader.samples_in_dataset()
                )
            )

        train_stats = Statistics(
            metric_names=self.train_metric_names, is_master_node=self.is_master_node
        )

        self.model.train()
        accum_freq = self.accum_freq if epoch >= self.accum_after_epoch else 1
        max_norm = getattr(self.opts, "common.grad_clip", None)

        # set the gradient to zero or None
        self._zero_grad()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        grad_norm = 0.0
        for batch_id, batch in enumerate(self.train_loader):
            if self.train_iterations > self.max_iterations:
                self.max_iterations_reached = True
                return -1, -1

            # move to device
            batch = move_to_device(opts=self.opts, x=batch, device=self.device)
            # apply mix-up transforms if any
            batch = self.apply_mixup_transforms(data=batch)

            batch_load_toc = time.time() - batch_load_start

            input_img, target_label = batch["image"], batch["label"]

            batch_size = self._get_batch_size(input_img)

            # update the learning rate
            self.optimizer = self.scheduler.update_lr(
                optimizer=self.optimizer, epoch=epoch, curr_iter=self.train_iterations
            )

            # adjust bn momentum
            if self.adjust_norm_mom is not None:
                self.adjust_norm_mom.adjust_momentum(
                    model=self.model, epoch=epoch, iteration=self.train_iterations
                )

            with autocast(enabled=self.mixed_precision_training):
                # prediction
                pred_label = self.model(input_img)
                # compute loss
                loss = self.criteria(
                    input_sample=input_img, prediction=pred_label, target=target_label
                )

                if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                    logger.error("Nan encountered in the loss.")

            # perform the backward pass with gradient accumulation [Optional]
            self.gradient_scalar.scale(loss).backward()

            if (batch_id + 1) % accum_freq == 0:
                if max_norm is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    self.gradient_scalar.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )

                if "grad_norm" in self.train_metric_names:
                    # compute grad_norm for logging purposes.
                    # We can't use the output of clip_grad_norm_ because it returns the total norm before clipping
                    grad_norm = self.compute_grad_norm()

                # optimizer step
                self.gradient_scalar.step(optimizer=self.optimizer)
                # update the scale for next batch
                self.gradient_scalar.update()
                # set the gradient to zero or None
                self._zero_grad()

                self.train_iterations += 1

                if self.model_ema is not None:
                    self.model_ema.update_parameters(self.model)

            metrics = metric_monitor(
                self.opts,
                pred_label=pred_label,
                target_label=target_label,
                loss=loss,
                grad_norm=grad_norm,
                use_distributed=self.use_distributed,
                metric_names=self.train_metric_names,
            )

            train_stats.update(
                metric_vals=metrics, batch_time=batch_load_toc, n=batch_size
            )

            if batch_id % self.log_freq == 0 and self.is_master_node:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(
                    epoch=epoch,
                    n_processed_samples=self.train_iterations,
                    total_samples=self.max_iterations,
                    learning_rate=lr,
                    elapsed_time=epoch_start_time,
                )

            batch_load_start = time.time()

        avg_loss = train_stats.avg_statistics(metric_name="loss")
        train_stats.epoch_summary(epoch=epoch, stage="training")
        avg_ckpt_metric = train_stats.avg_statistics(metric_name=self.ckpt_metric)

        gc.collect()

        return avg_loss, avg_ckpt_metric

    def val_epoch(self, epoch, model, extra_str=""):
        if self.val_loader is None:
            return 0.0, 0.0

        time.sleep(2)  # To prevent possible deadlock during epoch transition
        validation_stats = Statistics(
            metric_names=self.val_metric_names, is_master_node=self.is_master_node
        )

        if "coco_map" in self.val_metric_names:
            from metrics.coco_map import COCOEvaluator

            coco_evaluator = COCOEvaluator(
                opts=self.opts, iou_types=["bbox"], use_distributed=self.use_distributed
            )
        else:
            coco_evaluator = None

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.val_loader)
            processed_samples = 0
            lr = self.scheduler.retrieve_lr(self.optimizer)
            for batch_id, batch in enumerate(self.val_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                input_img, target_label = batch["image"], batch["label"]

                batch_size = self._get_batch_size(input_img)

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)
                    # compute loss
                    loss = self.criteria(
                        input_sample=input_img,
                        prediction=pred_label,
                        target=target_label,
                    )

                processed_samples += batch_size

                metrics = metric_monitor(
                    self.opts,
                    pred_label=pred_label,
                    target_label=target_label,
                    loss=loss,
                    use_distributed=self.use_distributed,
                    metric_names=self.val_metric_names,
                    is_evaluation=True,
                )

                validation_stats.update(
                    metric_vals=metrics, batch_time=0.0, n=batch_size
                )

                if coco_evaluator is not None:
                    coco_evaluator.prepare_predictions(
                        predictions=pred_label, targets=target_label
                    )

                if batch_id % self.log_freq == 0 and self.is_master_node:
                    validation_stats.iter_summary(
                        epoch=epoch,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=lr,
                    )

        validation_stats.epoch_summary(epoch=epoch, stage="validation" + extra_str)
        avg_loss = validation_stats.avg_statistics(metric_name="loss")
        avg_ckpt_metric = validation_stats.avg_statistics(metric_name=self.ckpt_metric)

        if coco_evaluator is not None:
            # synchronize across different processes and aggregate the results
            coco_evaluator.gather_coco_results()
            coco_map = coco_evaluator.summarize_coco_results()

            if self.ckpt_metric == "coco_map" and "bbox" in coco_map:
                avg_ckpt_metric = round(coco_map["bbox"], 5)

        if avg_ckpt_metric is None:
            avg_ckpt_metric = avg_loss

        gc.collect()

        return avg_loss, avg_ckpt_metric

    def find_easy_samples(self, epoch, model, *args, **kwargs):
        """
        This function identifies easy samples in the training set and removes them from training.

        .. note::
            Currently, this is implemented separately to avoid breaking the training and validation pipeline. In future,
            this will be combined with main training loop to reduce overhead.
        """

        time.sleep(2)  # To prevent possible deadlock during epoch transition

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        if self.is_master_node:
            logger.log("Trying to find easy samples in epoch {}".format(epoch))

        with torch.no_grad():
            easy_sample_ids_tensor = torch.zeros_like(self.running_sum_tensor)

            for batch_id, batch in enumerate(self.train_loader_set):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                input_img, target_label = batch["image"], batch["label"]

                sample_ids = None
                if "sample_id" in batch:
                    sample_ids = batch["sample_id"]
                else:
                    self.sample_efficient_training = False
                    if self.is_master_node:
                        logger.log(
                            "Sample Ids are required in a batch for sample efficient training. "
                            "sample_id key not found in batch. Disabling sample efficient training."
                        )
                    break

                if sample_ids is None:
                    logger.log("Sample Ids can't be none")
                    break

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)
                    pred_label = F.softmax(pred_label, dim=-1)

                pred_conf, pred_indices = torch.max(pred_label, dim=-1)

                easy_samples = torch.logical_and(
                    pred_indices.eq(
                        target_label
                    ),  # condition 1: Predicted label == Target label
                    pred_conf
                    >= self.sample_confidence,  # condition 2: prediction confidence >= desired confidence
                )

                if easy_samples.numel() > 0:
                    easy_sample_ids = sample_ids[easy_samples]
                    # find easy samples as per condition 1 and 2 and set their values to 1
                    easy_sample_ids_tensor[easy_sample_ids] = 1

            # synchronize tensors
            if self.use_distributed:
                # sync across all GPUs.
                easy_sample_ids_tensor = reduce_tensor_sum(easy_sample_ids_tensor)

            # some samples which are classified easy earlier may have been classified hard now.
            easy_sample_ids_tensor[easy_sample_ids_tensor == 0] = -1

            if self.is_master_node:
                logger.debug(
                    "Number of easy samples found during epoch {} are {}".format(
                        epoch,
                        easy_sample_ids_tensor[easy_sample_ids_tensor > 0].sum().item(),
                    )
                )

            self.running_sum_tensor = torch.clip(
                self.running_sum_tensor + easy_sample_ids_tensor,
                min=0,
                max=self.min_sample_frequency,
            )

            if self.running_sum_tensor.sum() > 0:
                skip_sample_ids = (
                    self.running_sum_tensor >= self.min_sample_frequency
                ).nonzero(as_tuple=True)[0]

                if skip_sample_ids.numel() > 0:
                    skip_samples = skip_sample_ids.cpu().numpy().tolist()

                    new_sample_ids = [
                        s_id
                        for s_id in self.sample_ids_orig
                        if s_id not in skip_sample_ids
                    ]

                    # update the train loader indices
                    self.train_loader.update_indices(new_sample_ids)

                    if self.is_master_node:
                        logger.debug(
                            "Number of samples to skip after epoch {} are {}".format(
                                epoch, len(skip_samples)
                            )
                        )

    @staticmethod
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

    def run(self, train_sampler=None):
        if train_sampler is None and self.is_master_node:
            logger.error("Train sampler cannot be None")

        copy_at_epoch = getattr(self.opts, "ema.copy_at_epoch", -1)
        train_start_time = time.time()
        save_dir = getattr(self.opts, "common.exp_loc", "results")

        cfg_file = getattr(self.opts, "common.config_file", None)
        if cfg_file is not None and self.is_master_node:
            dst_cfg_file = "{}/config.yaml".format(save_dir)
            shutil.copy(src=cfg_file, dst=dst_cfg_file)
            logger.info(
                "Configuration file is stored here: {}".format(
                    logger.color_text(dst_cfg_file)
                )
            )

        keep_k_best_ckpts = getattr(self.opts, "common.k_best_checkpoints", 5)
        ema_best_metric = self.best_metric
        is_ema_best = False

        try:
            max_epochs = getattr(self.opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
            max_checkpoint_metric = getattr(
                self.opts, "stats.checkpoint_metric_max", False
            )
            for epoch in range(self.start_epoch, max_epochs):
                # Note that we are using our owm implementations of data samplers
                # and we have defined this function for both distributed and non-distributed cases
                train_sampler.set_epoch(epoch)
                train_sampler.update_scales(
                    epoch=epoch, is_master_node=self.is_master_node
                )

                train_loss, train_ckpt_metric = self.train_epoch(epoch)

                val_loss, val_ckpt_metric = self.val_epoch(
                    epoch=epoch, model=self.model
                )

                if epoch == copy_at_epoch and self.model_ema is not None:
                    if self.is_master_node:
                        logger.log("Copying EMA weights")
                    # copy model_src weights to model_tgt
                    self.model = copy_weights(
                        model_tgt=self.model, model_src=self.model_ema
                    )
                    if self.is_master_node:
                        logger.log("EMA weights copied")
                        logger.log("Running validation after Copying EMA model weights")
                    self.val_epoch(epoch=epoch, model=self.model)

                if max_checkpoint_metric:
                    is_best = val_ckpt_metric >= self.best_metric
                    self.best_metric = max(val_ckpt_metric, self.best_metric)
                else:
                    is_best = val_ckpt_metric <= self.best_metric
                    self.best_metric = min(val_ckpt_metric, self.best_metric)

                val_ema_loss = None
                val_ema_ckpt_metric = None
                if self.model_ema is not None:
                    val_ema_loss, val_ema_ckpt_metric = self.val_epoch(
                        epoch=epoch, model=self.model_ema.ema_model, extra_str=" (EMA)"
                    )
                    if max_checkpoint_metric:
                        is_ema_best = val_ema_ckpt_metric >= ema_best_metric
                        ema_best_metric = max(val_ema_ckpt_metric, ema_best_metric)
                    else:
                        is_ema_best = val_ema_ckpt_metric <= ema_best_metric
                        ema_best_metric = min(val_ema_ckpt_metric, ema_best_metric)

                # sample efficient training
                if (
                    self.sample_efficient_training
                    and (epoch + 1) % self.find_easy_samples_every_k_epoch == 0
                ):
                    self.find_easy_samples(
                        epoch=epoch,
                        model=self.model
                        if self.model_ema is not None
                        else self.model_ema.ema_model,
                    )

                gc.collect()

                if self.is_master_node:
                    save_checkpoint(
                        iterations=self.train_iterations,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        best_metric=self.best_metric,
                        is_best=is_best,
                        save_dir=save_dir,
                        model_ema=self.model_ema,
                        is_ema_best=is_ema_best,
                        ema_best_metric=ema_best_metric,
                        gradient_scalar=self.gradient_scalar,
                        max_ckpt_metric=max_checkpoint_metric,
                        k_best_checkpoints=keep_k_best_ckpts,
                        save_all_checkpoints=self.save_all_checkpoints,
                    )
                    logger.info(
                        "Checkpoints saved at: {}".format(save_dir), print_line=True
                    )

                if self.is_master_node:
                    lr_list = self.scheduler.retrieve_lr(self.optimizer)

                    if self.tb_log_writer is not None:
                        self.log_metrics(
                            lrs=lr_list,
                            log_writer=self.tb_log_writer,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            epoch=epoch,
                            best_metric=self.best_metric,
                            val_ema_loss=val_ema_loss,
                            ckpt_metric_name=self.ckpt_metric,
                            train_ckpt_metric=train_ckpt_metric,
                            val_ckpt_metric=val_ckpt_metric,
                            val_ema_ckpt_metric=val_ema_ckpt_metric,
                        )
                    if self.bolt_log_writer is not None:
                        self.log_metrics(
                            lrs=lr_list,
                            log_writer=self.bolt_log_writer,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            epoch=epoch,
                            best_metric=self.best_metric,
                            val_ema_loss=val_ema_loss,
                            ckpt_metric_name=self.ckpt_metric,
                            train_ckpt_metric=train_ckpt_metric,
                            val_ckpt_metric=val_ckpt_metric,
                            val_ema_ckpt_metric=val_ema_ckpt_metric,
                        )

                if self.max_iterations_reached:
                    if self.use_distributed:
                        dist_barrier()

                    if self.is_master_node:
                        logger.info("Max. iterations for training reached")
                    break

        except KeyboardInterrupt as e:
            if self.is_master_node:
                logger.log("Keyboard interruption. Exiting from early training")
                raise e
        except Exception as e:
            if "out of memory" in str(e):
                logger.log("OOM exception occured")
                n_gpus = getattr(self.opts, "dev.num_gpus", 1)
                for dev_id in range(n_gpus):
                    mem_summary = torch.cuda.memory_summary(
                        device=torch.device("cuda:{}".format(dev_id)), abbreviated=True
                    )
                    logger.log("Memory summary for device id: {}".format(dev_id))
                    print(mem_summary)
            else:
                logger.log(
                    "Exception occurred that interrupted the training. {}".format(
                        str(e)
                    )
                )
                print(e)
                traceback.print_exc()
                raise e
        finally:
            use_distributed = getattr(self.opts, "ddp.use_distributed", False)
            if use_distributed:
                torch.distributed.destroy_process_group()

            torch.cuda.empty_cache()

            if self.is_master_node and self.tb_log_writer is not None:
                self.tb_log_writer.close()

            if self.is_master_node:
                train_end_time = time.time()
                hours, rem = divmod(train_end_time - train_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours), int(minutes), seconds
                )
                logger.log("Training took {}".format(train_time_str))

            try:
                exit(0)
            except Exception as e:
                pass
            finally:
                pass

    def run_loss_landscape(self):
        # Loss landscape code is adapted from https://github.com/xxxnell/how-do-vits-work
        ll_start_time = time.time()
        try:
            n_points = getattr(self.opts, "loss_landscape.n_points", 32)
            min_x = getattr(self.opts, "loss_landscape.min_x", -1.0)
            max_x = getattr(self.opts, "loss_landscape.max_x", 1.0)
            min_y = getattr(self.opts, "loss_landscape.min_y", -1.0)
            max_y = getattr(self.opts, "loss_landscape.max_y", 1.0)

            if self.is_master_node:
                logger.log(
                    "Loss landscape coord space params: \n\tmin_x={}\n\tmax_x={}\n\tmin_y={}\n\tmax_y={}\n\tn_points={}".format(
                        min_x, max_x, min_y, max_y, n_points
                    )
                )

            ll_metrics = ["loss"]
            ll_stats = Statistics(
                metric_names=ll_metrics, is_master_node=self.is_master_node
            )
            has_module = hasattr(self.model, "module")
            model_name = (
                self.model.module.__class__.__name__
                if has_module
                else self.model.__class__.__name__
            )
            save_dir = getattr(self.opts, "common.exp_loc", "results")

            # copy the model and create bases
            model = copy.deepcopy(self.model)
            weight_state_0 = (
                copy.deepcopy(model.module.state_dict())
                if has_module
                else copy.deepcopy(model.state_dict())
            )
            bases = ll_utils.create_bases(
                model=model, device=self.device, has_module=has_module
            )

            xs = np.linspace(min_x, max_x, n_points)
            ys = np.linspace(min_y, max_y, n_points)

            grid_a, grid_b = np.meshgrid(xs, ys, indexing="xy")
            loss_surface = np.empty_like(grid_a)

            epoch = -1
            for coord_a, coord_b in product(range(n_points), range(n_points)):
                epoch += 1
                coords_list = [grid_a[coord_a, coord_b], grid_b[coord_a, coord_b]]
                weight_state_1 = copy.deepcopy(weight_state_0)
                gs = [{k: r * bs[k] for k in bs} for r, bs in zip(coords_list, bases)]
                gs = {
                    k: torch.sum(torch.stack([g[k] for g in gs]), dim=0)
                    + weight_state_1[k]
                    for k in gs[0]
                }

                # load the weights
                model.module.load_state_dict(
                    gs
                ) if has_module else model.load_state_dict(gs)

                model = model.to(device=self.device)
                model.eval()

                total_samples = len(self.val_loader)
                with torch.no_grad():
                    epoch_start_time = time.time()
                    processed_samples = 0
                    for batch_id, batch in enumerate(self.val_loader):
                        batch = move_to_device(
                            opts=self.opts, x=batch, device=self.device
                        )
                        input_img, target_label = batch["image"], batch["label"]

                        batch_size = self._get_batch_size(x=input_img)
                        processed_samples += batch_size

                        # make the prediction and compute loss
                        pred_label = model(input_img)
                        loss = self.criteria(
                            input_sample=input_img,
                            prediction=pred_label,
                            target=target_label,
                        )

                        if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                            logger.error("Nan encountered in the loss.")

                        metrics = metric_monitor(
                            self.opts,
                            pred_label=pred_label,
                            target_label=target_label,
                            loss=loss,
                            use_distributed=self.use_distributed,
                            metric_names=ll_metrics,
                            is_evaluation=True,
                        )

                        ll_stats.update(
                            metric_vals=metrics, batch_time=0.0, n=batch_size
                        )

                        if batch_id % self.log_freq == 0 and self.is_master_node:
                            ll_stats.iter_summary(
                                epoch=epoch,
                                n_processed_samples=processed_samples,
                                total_samples=total_samples,
                                elapsed_time=epoch_start_time,
                                learning_rate=0.0,
                            )

                    avg_loss = ll_stats.avg_statistics(metric_name="loss")
                    loss_surface[coord_a, coord_b] = avg_loss
                    if self.is_master_node:
                        print(
                            "x: {:.2f}, y: {:.2f}, loss: {:.2f}".format(
                                coords_list[0], coords_list[1], avg_loss
                            )
                        )

                    if self.is_master_node:
                        lr_list = [0.0]

                        if self.tb_log_writer is not None:
                            self.log_metrics(
                                lrs=lr_list,
                                log_writer=self.tb_log_writer,
                                train_loss=0.0,
                                val_loss=avg_loss,
                                epoch=epoch,
                                best_metric=self.best_metric,
                                val_ema_loss=None,
                                ckpt_metric_name=None,
                                train_ckpt_metric=None,
                                val_ckpt_metric=None,
                                val_ema_ckpt_metric=None,
                            )
                        if self.bolt_log_writer is not None:
                            self.log_metrics(
                                lrs=lr_list,
                                log_writer=self.bolt_log_writer,
                                train_loss=0.0,
                                val_loss=avg_loss,
                                epoch=epoch,
                                best_metric=self.best_metric,
                                val_ema_loss=None,
                                ckpt_metric_name=None,
                                train_ckpt_metric=None,
                                val_ckpt_metric=None,
                                val_ema_ckpt_metric=None,
                            )

                    gc.collect()
                    # take a small nap
                    time.sleep(1)

            if self.is_master_node:
                ll_utils.plot_save_graphs(
                    save_dir=save_dir,
                    model_name=model_name,
                    grid_a=grid_a,
                    grid_b=grid_b,
                    loss_surface=loss_surface,
                    resolution=n_points,
                )
        except KeyboardInterrupt as e:
            if self.is_master_node:
                logger.log("Keyboard interruption. Exiting from early training")
                raise e
        except Exception as e:
            if "out of memory" in str(e):
                logger.log("OOM exception occured")
                n_gpus = getattr(self.opts, "dev.num_gpus", 1)
                for dev_id in range(n_gpus):
                    mem_summary = torch.cuda.memory_summary(
                        device=torch.device("cuda:{}".format(dev_id)), abbreviated=True
                    )
                    logger.log("Memory summary for device id: {}".format(dev_id))
                    print(mem_summary)
            else:
                logger.log(
                    "Exception occurred that interrupted the training. {}".format(
                        str(e)
                    )
                )
                print(e)
                raise e
        finally:
            if self.use_distributed:
                torch.distributed.destroy_process_group()

            torch.cuda.empty_cache()

            if self.is_master_node:
                ll_end_time = time.time()
                hours, rem = divmod(ll_end_time - ll_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours), int(minutes), seconds
                )
                logger.log("Loss landspace evaluation took {}".format(train_time_str))

            try:
                exit(0)
            except Exception as e:
                pass
            finally:
                pass

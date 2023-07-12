#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import copy
import gc
import shutil
import time
import traceback
from itertools import product
from typing import Dict, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from common import DEFAULT_EPOCHS, DEFAULT_ITERATIONS, DEFAULT_LOG_FREQ, if_test_env
from data.transforms.image_torch import apply_mixing_transforms
from engine.utils import autocast_fn, get_batch_size, get_log_writers, log_metrics
from loss_landscape import landscape_utils as ll_utils
from metrics.stats import Statistics
from options.parse_args import parse_validation_metric_names
from utils import logger
from utils.checkpoint_utils import (
    copy_weights,
    save_checkpoint,
    save_interval_checkpoint,
)
from utils.common_utils import move_to_device, unwrap_model_fn
from utils.ddp_utils import dist_barrier, is_master
from utils.tensor_utils import reduce_tensor_sum


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
        gradient_scaler,
        start_epoch: int = 0,
        start_iteration: int = 0,
        best_metric: float = 0.0,
        model_ema=None,
        *args,
        **kwargs,
    ) -> None:
        super(Trainer, self).__init__()

        self.opts = opts

        self.model = model
        self.model_ema = model_ema
        self.criteria = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_scaler = gradient_scaler

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
        self.mixed_precision_dtype = getattr(
            opts, "common.mixed_precision_dtype", "float16"
        )

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

        (
            self.val_metric_names,
            self.ckpt_metric,
            self.ckpt_submetric,
        ) = parse_validation_metric_names(self.opts)

        self.save_all_checkpoints = getattr(
            self.opts, "common.save_all_checkpoints", False
        )

        self.save_location = getattr(opts, "common.exp_loc", "results/run_1")

        self.log_writers = get_log_writers(self.opts, save_location=self.save_location)

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

        save_interval_freq = getattr(opts, "common.save_interval_freq", 0)
        # save interval checkpoints every `save_interval_freq` updates on the master node
        self.save_interval = self.is_master_node and save_interval_freq > 0
        self.save_interval_freq = save_interval_freq

    def compute_grad_norm(self):
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None

        norm_type = 2.0  # L2 norm

        inv_scale = 1.0 / self.gradient_scaler.get_scale()
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

    def _zero_grad(self):
        if self.set_grad_to_none:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad()

    def train_epoch(self, epoch):
        time.sleep(
            if_test_env(0.5, otherwise=2)
        )  # To prevent possible deadlock during epoch transition

        if self.is_master_node:
            logger.double_dash_line()
            logger.debug(
                "Training epoch {} with {} samples".format(
                    epoch, self.train_loader.samples_in_dataset()
                )
            )

        train_stats = Statistics(
            opts=self.opts,
            metric_names=self.train_metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        self.model.train()
        # criteria is also a nn.Module and we may need access to training property in some
        # loss functions. So, to enable, that, we set criteria to train/eval mode
        self.criteria.train()

        accum_freq = self.accum_freq if epoch >= self.accum_after_epoch else 1
        max_norm = getattr(self.opts, "common.grad_clip", None)

        # set the gradient to zero or None
        self._zero_grad()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        grad_norm = torch.tensor([0.0], dtype=torch.float, device=self.device)
        for batch_id, batch in enumerate(self.train_loader):
            if self.train_iterations > self.max_iterations:
                self.max_iterations_reached = True
                return -1, -1

            # move to device
            batch = move_to_device(opts=self.opts, x=batch, device=self.device)
            # apply mix-up transforms if any
            batch = apply_mixing_transforms(opts=self.opts, data=batch)

            batch_load_toc = time.time() - batch_load_start

            samples, targets = batch["samples"], batch["targets"]

            batch_size = get_batch_size(samples)

            # update the learning rate
            self.optimizer = self.scheduler.update_lr(
                optimizer=self.optimizer, epoch=epoch, curr_iter=self.train_iterations
            )

            # adjust bn momentum
            if self.adjust_norm_mom is not None:
                self.adjust_norm_mom.adjust_momentum(
                    model=self.model, epoch=epoch, iteration=self.train_iterations
                )

            with autocast_fn(
                enabled=self.mixed_precision_training,
                amp_precision=self.mixed_precision_dtype,
            ):
                # prediction
                pred_label = self.model(samples)
                # compute loss
                loss_dict_or_tensor: Union[Dict, Tensor] = self.criteria(
                    input_sample=samples,
                    prediction=pred_label,
                    target=targets,
                    epoch=epoch,
                    iterations=self.train_iterations,
                )

                if isinstance(loss_dict_or_tensor, Dict):
                    if "total_loss" not in loss_dict_or_tensor.keys():
                        logger.error(
                            "total_loss key is required for loss functions that return outputs as dictionary."
                        )
                    loss = loss_dict_or_tensor["total_loss"]
                elif isinstance(loss_dict_or_tensor, Tensor):
                    loss = loss_dict_or_tensor
                else:
                    logger.error("Loss value should be an instance of Tensor or Dict")

                if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                    logger.error("Nan encountered in the loss.")

            # perform the backward pass with gradient accumulation [Optional]
            self.gradient_scaler.scale(loss).backward()

            if (batch_id + 1) % accum_freq == 0:
                if max_norm is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    self.gradient_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )

                if "grad_norm" in self.train_metric_names:
                    # compute grad_norm for logging purposes.
                    # We can't use the output of clip_grad_norm_ because it returns the total norm before clipping
                    grad_norm = self.compute_grad_norm()

                # optimizer step
                self.gradient_scaler.step(optimizer=self.optimizer)
                # update the scale for next batch
                self.gradient_scaler.update()
                # set the gradient to zero or None
                self._zero_grad()

                self.train_iterations += 1

                if self.model_ema is not None:
                    self.model_ema.update_parameters(self.model)

            train_stats.update(
                pred_label=pred_label,
                target_label=targets,
                extras={"loss": loss_dict_or_tensor, "grad_norm": grad_norm},
                batch_time=batch_load_toc,
                batch_size=batch_size,
            )

            # save the checkpoint every N updates
            if (
                self.save_interval
                and (self.train_iterations % self.save_interval_freq) == 0
            ):

                save_interval_checkpoint(
                    iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    best_metric=loss.item(),
                    save_dir=self.save_location,
                    gradient_scaler=self.gradient_scaler,
                    model_ema=self.model_ema,
                )
                logger.info(
                    "Checkpoints saved after {} updates at: {}".format(
                        self.train_iterations, self.save_location
                    ),
                    print_line=True,
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

        avg_loss = train_stats.avg_statistics(
            metric_name="loss", sub_metric_name="total_loss"
        )
        train_stats.epoch_summary(epoch=epoch, stage="training")
        avg_ckpt_metric = train_stats.avg_statistics(
            metric_name=self.ckpt_metric, sub_metric_name=self.ckpt_submetric
        )

        gc.collect()

        return avg_loss, avg_ckpt_metric

    def val_epoch(self, epoch, model, extra_str=""):
        if self.val_loader is None:
            return 0.0, 0.0

        time.sleep(
            if_test_env(0.5, otherwise=2)
        )  # To prevent possible deadlock during epoch transition
        validation_stats = Statistics(
            opts=self.opts,
            metric_names=self.val_metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()
        # criteria is also a nn.Module and we may need access to training property in some
        # loss functions. So, to enable, that, we set criteria to train/eval mode
        self.criteria.eval()

        if model.training:
            if self.is_master_node:
                logger.warning(
                    "Model is in training mode. Switching to evaluation mode"
                )
            model.eval()

        if self.criteria.training:
            self.criteria.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.val_loader)
            processed_samples = 0
            lr = self.scheduler.retrieve_lr(self.optimizer)
            for batch_id, batch in enumerate(self.val_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]

                batch_size = get_batch_size(samples)

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)
                    # compute loss
                    loss_dict_or_tensor = self.criteria(
                        input_sample=samples,
                        prediction=pred_label,
                        target=targets,
                    )

                processed_samples += batch_size

                validation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={"loss": loss_dict_or_tensor},
                    batch_time=0.0,
                    batch_size=batch_size,
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
        avg_loss = validation_stats.avg_statistics(
            metric_name="loss", sub_metric_name="total_loss"
        )
        avg_ckpt_metric = validation_stats.avg_statistics(
            metric_name=self.ckpt_metric, sub_metric_name=self.ckpt_submetric
        )

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

        time.sleep(
            if_test_env(0.5, otherwise=2)
        )  # To prevent possible deadlock during epoch transition

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

                samples, targets = batch["samples"], batch["targets"]

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

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)
                    pred_label = F.softmax(pred_label, dim=-1)

                pred_conf, pred_indices = torch.max(pred_label, dim=-1)

                easy_samples = torch.logical_and(
                    pred_indices.eq(
                        targets
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

    def run(self, train_sampler=None):
        if train_sampler is None and self.is_master_node:
            logger.error("Train sampler cannot be None")

        copy_at_epoch = getattr(self.opts, "ema.copy_at_epoch", -1)
        train_start_time = time.time()

        cfg_file = getattr(self.opts, "common.config_file", None)
        if cfg_file is not None and self.is_master_node:
            dst_cfg_file = "{}/config.yaml".format(self.save_location)
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
                        save_dir=self.save_location,
                        model_ema=self.model_ema,
                        is_ema_best=is_ema_best,
                        ema_best_metric=ema_best_metric,
                        gradient_scaler=self.gradient_scaler,
                        max_ckpt_metric=max_checkpoint_metric,
                        k_best_checkpoints=keep_k_best_ckpts,
                        save_all_checkpoints=self.save_all_checkpoints,
                    )
                    logger.info(
                        "Checkpoints saved at: {}".format(self.save_location),
                        print_line=True,
                    )

                if self.is_master_node:
                    lr_list = self.scheduler.retrieve_lr(self.optimizer)

                    for log_writer in self.log_writers:
                        log_metrics(
                            lrs=lr_list,
                            log_writer=log_writer,
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

            logger.log(
                f"Exception occurred that interrupted the training:\n{traceback.format_exc()}"
            )
            raise e
        finally:
            use_distributed = getattr(self.opts, "ddp.use_distributed", False)
            if use_distributed:
                torch.distributed.destroy_process_group()

            torch.cuda.empty_cache()

            for log_writer in self.log_writers:
                log_writer.close()

            if self.is_master_node:
                train_end_time = time.time()
                hours, rem = divmod(train_end_time - train_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours), int(minutes), seconds
                )
                logger.log("Training took {}".format(train_time_str))

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
                opts=self.opts,
                metric_names=ll_metrics,
                is_master_node=self.is_master_node,
                is_distributed=self.use_distributed,
                log_writers=self.log_writers,
            )
            has_module = hasattr(self.model, "module")
            unwrapped_model = unwrap_model_fn(self.model)
            model_name = unwrapped_model.__class__.__name__

            # copy the model and create bases
            model = copy.deepcopy(self.model)
            weight_state_0 = unwrapped_model.state_dict()
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
                unwrapped_model.load_state_dict(gs)

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
                        samples, targets = batch["samples"], batch["targets"]

                        batch_size = get_batch_size(samples)
                        processed_samples += batch_size

                        # make the prediction and compute loss
                        pred_label = model(samples)
                        loss_dict_or_tensor: Union[Dict, Tensor] = self.criteria(
                            input_sample=samples,
                            prediction=pred_label,
                            target=targets,
                        )

                        if isinstance(loss_dict_or_tensor, Dict):
                            if "total_loss" not in loss_dict_or_tensor.keys():
                                logger.error(
                                    "total_loss key is required for loss functions that return outputs as dictionary."
                                )
                            loss = loss_dict_or_tensor["total_loss"]
                        elif isinstance(loss_dict_or_tensor, Tensor):
                            loss = loss_dict_or_tensor
                        else:
                            logger.error(
                                "Loss value should be an instance of Tensor or Dict"
                            )

                        if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                            logger.error("Nan encountered in the loss.")

                        ll_stats.update(
                            pred_label=pred_label,
                            target_label=targets,
                            extras={"loss": loss_dict_or_tensor},
                            batch_time=0.0,
                            batch_size=batch_size,
                        )

                        if batch_id % self.log_freq == 0 and self.is_master_node:
                            ll_stats.iter_summary(
                                epoch=epoch,
                                n_processed_samples=processed_samples,
                                total_samples=total_samples,
                                elapsed_time=epoch_start_time,
                                learning_rate=0.0,
                            )

                    avg_loss = ll_stats.avg_statistics(
                        metric_name="loss", sub_metric_name="total_loss"
                    )
                    loss_surface[coord_a, coord_b] = avg_loss
                    if self.is_master_node:
                        print(
                            "x: {:.2f}, y: {:.2f}, loss: {:.2f}".format(
                                coords_list[0], coords_list[1], avg_loss
                            )
                        )

                    if self.is_master_node:
                        lr_list = [0.0]

                        for log_writer in self.log_writers:
                            log_metrics(
                                lrs=lr_list,
                                log_writer=log_writer,
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
                    time.sleep(if_test_env(0, otherwise=1))

            if self.is_master_node:
                ll_utils.plot_save_graphs(
                    save_dir=self.save_location,
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

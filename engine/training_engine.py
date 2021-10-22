#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
import gc
from torch.cuda.amp import autocast
from utils import logger
from utils.common_utils import create_directories
import time
import shutil
from typing import Dict

from engine.utils import print_summary
from utils.ddp_utils import is_master
from utils.checkpoint_utils import copy_weights, save_checkpoint
from metrics import Statistics, metric_monitor
from common import DEFAULT_ITERATIONS, DEFAULT_EPOCHS, DEFAULT_LOG_FREQ

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    SummaryWriter = None


class Trainer(object):
    """
        This class defines the training and validation code for training models with CVNets
    """
    def __init__(self, opts,
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
                 *args, **kwargs) -> None:
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
        self.max_iterations = getattr(self.opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)
        self.accum_freq = getattr(self.opts, "common.accum_freq", 1)
        self.accum_after_epoch = getattr(self.opts, "common.accum_after_epoch", 0)

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        self.metric_names = getattr(opts, "stats.name", ['loss'])
        if isinstance(self.metric_names, str):
            self.metric_names = [self.metric_names]
        assert isinstance(self.metric_names, list), "Type of metric names should be list. Got: {}".format(
            type(self.metric_names))

        if 'loss' not in self.metric_names:
            self.metric_names.append(self.metric_names)

        self.ckpt_metric = getattr(self.opts, "stats.checkpoint_metric", "loss")
        assert self.ckpt_metric in self.metric_names, \
            "Checkpoint metric should be part of metric names. Metric names: {}, Checkpoint metric: {}".format(
                self.metric_names, self.ckpt_metric)
        self.ckpt_metric = self.ckpt_metric.lower()

        self.tb_log_writter = None
        if SummaryWriter is not None and self.is_master_node:
            self.setup_log_writer()

        if self.is_master_node:
            print_summary(opts=self.opts,
                          model=self.model,
                          criteria=self.criteria,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler)

        self.adjust_norm_mom = None
        if getattr(opts, "adjust_bn_momentum.enable", True):
            from cvnets.layers import AdjustBatchNormMomentum
            self.adjust_norm_mom = AdjustBatchNormMomentum(opts=opts)
            if self.is_master_node:
                logger.log("Batch normalization momentum will be annealed during training.")
                print(self.adjust_norm_mom)

    def setup_log_writer(self):
        exp_dir = getattr(self.opts, "common.exp_loc", "results/run_1")
        exp_dir = '{}/tb_logs'.format(exp_dir)
        create_directories(dir_path=exp_dir, is_master_node=self.is_master_node)
        self.tb_log_writter = SummaryWriter(log_dir=exp_dir, comment='Training and Validation logs')

    def train_epoch(self, epoch):
        time.sleep(2)  # To prevent possible deadlock during epoch transition
        train_stats = Statistics(metric_names=self.metric_names, is_master_node=self.is_master_node)

        self.model.train()
        accum_freq = self.accum_freq if epoch > self.accum_after_epoch else 1
        max_norm = getattr(self.opts, "common.grad_clip", None)

        self.optimizer.zero_grad()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        for batch_id, batch in enumerate(self.train_loader):
            if self.train_iterations > self.max_iterations:
                self.max_iterations_reached = True
                return -1, -1

            batch_load_toc = time.time() - batch_load_start
            input_img, target_label = batch['image'], batch['label']
            # move data to device
            input_img = input_img.to(self.device)
            if isinstance(target_label, Dict):
                for k, v in target_label.items():
                    target_label[k] = v.to(self.device)
            else:
                target_label = target_label.to(self.device)

            batch_size = input_img.shape[0]

            # update the learning rate
            self.optimizer = self.scheduler.update_lr(optimizer=self.optimizer, epoch=epoch,
                                                      curr_iter=self.train_iterations)

            # adjust bn momentum
            if self.adjust_norm_mom is not None:
                self.adjust_norm_mom.adjust_momentum(model=self.model,
                                                     epoch=epoch,
                                                     iteration=self.train_iterations)

            with autocast(enabled=self.mixed_precision_training):
                # prediction
                pred_label = self.model(input_img)
                # compute loss
                loss = self.criteria(input_sample=input_img, prediction=pred_label, target=target_label)

                if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                    import pdb
                    pdb.set_trace()

            # perform the backward pass with gradient accumulation [Optional]
            self.gradient_scalar.scale(loss).backward()

            if (batch_id + 1) % accum_freq == 0:
                if max_norm is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    self.gradient_scalar.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

                # optimizer step
                self.gradient_scalar.step(optimizer=self.optimizer)
                # update the scale for next batch
                self.gradient_scalar.update()
                # set grads to zero
                self.optimizer.zero_grad()

                if self.model_ema is not None:
                    self.model_ema.update_parameters(self.model)

            metrics = metric_monitor(pred_label=pred_label, target_label=target_label, loss=loss,
                                     use_distributed=self.use_distributed, metric_names=self.metric_names)

            train_stats.update(metric_vals=metrics, batch_time=batch_load_toc, n=batch_size)

            if batch_id % self.log_freq == 0 and self.is_master_node:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(epoch=epoch,
                                         n_processed_samples=self.train_iterations,
                                         total_samples=self.max_iterations,
                                         learning_rate=lr,
                                         elapsed_time=epoch_start_time)

            batch_load_start = time.time()
            self.train_iterations += 1

        avg_loss = train_stats.avg_statistics(metric_name='loss')
        train_stats.epoch_summary(epoch=epoch, stage="training")
        avg_ckpt_metric = train_stats.avg_statistics(metric_name=self.ckpt_metric)
        return avg_loss, avg_ckpt_metric

    def val_epoch(self, epoch, model, extra_str=""):
        time.sleep(2)  # To prevent possible deadlock during epoch transition
        validation_stats = Statistics(metric_names=self.metric_names, is_master_node=self.is_master_node)

        model.eval()
        if model.training and self.is_master_node:
            logger.warning('Model is in training mode. Switching to evaluation mode')
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.val_loader)
            processed_samples = 0
            lr = self.scheduler.retrieve_lr(self.optimizer)
            for batch_id, batch in enumerate(self.val_loader):
                input_img, target_label = batch['image'], batch['label']

                # move data to device
                input_img = input_img.to(self.device)

                if isinstance(target_label, Dict):
                    for k, v in target_label.items():
                        target_label[k] = v.to(self.device)
                else:
                    target_label = target_label.to(self.device)

                batch_size = input_img.shape[0]

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)
                    # compute loss
                    loss = self.criteria(input_sample=input_img, prediction=pred_label, target=target_label)

                processed_samples += batch_size

                metrics = metric_monitor(pred_label=pred_label, target_label=target_label, loss=loss,
                                         use_distributed=self.use_distributed, metric_names=self.metric_names)
                validation_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

                if batch_id % self.log_freq == 0 and self.is_master_node:
                    validation_stats.iter_summary(epoch=epoch,
                                                  n_processed_samples=processed_samples,
                                                  total_samples=total_samples,
                                                  elapsed_time=epoch_start_time,
                                                  learning_rate=lr
                                                  )

        validation_stats.epoch_summary(epoch=epoch, stage="validation" + extra_str)
        avg_loss = validation_stats.avg_statistics(metric_name='loss')
        avg_ckpt_metric = validation_stats.avg_statistics(metric_name=self.ckpt_metric)
        return avg_loss, avg_ckpt_metric

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
            logger.info('Configuration file is stored here: {}'.format(logger.color_text(dst_cfg_file)))

        keep_k_best_ckpts = getattr(self.opts, "common.k_best_checkpoints", 5)
        ema_best_metric = self.best_metric
        is_ema_best = False
        try:
            max_epochs = getattr(self.opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
            for epoch in range(self.start_epoch, max_epochs):
                # Note that we are using our owm implementations of data samplers
                # and we have defined this function for both distributed and non-distributed cases
                train_sampler.set_epoch(epoch)
                train_sampler.update_scales(epoch=epoch, is_master_node=self.is_master_node)

                train_loss, train_ckpt_metric = self.train_epoch(epoch)

                val_loss, val_ckpt_metric = self.val_epoch(epoch=epoch, model=self.model)

                if epoch == copy_at_epoch and self.model_ema is not None:
                    if self.is_master_node:
                        logger.log('Copying EMA weights')
                    # copy model_src weights to model_tgt
                    self.model = copy_weights(model_tgt=self.model, model_src=self.model_ema)
                    if self.is_master_node:
                        logger.log('EMA weights copied')
                        logger.log('Running validation after Copying EMA model weights')
                    self.val_epoch(epoch=epoch, model=self.model)

                gc.collect()

                max_checkpoint_metric = getattr(self.opts, "stats.checkpoint_metric_max", False)
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
                        epoch=epoch,
                        model=self.model_ema.ema_model,
                        extra_str=" (EMA)"
                    )
                    if max_checkpoint_metric:
                        is_ema_best = val_ema_ckpt_metric >= ema_best_metric
                        ema_best_metric = max(val_ema_ckpt_metric, ema_best_metric)
                    else:
                        is_ema_best = val_ema_ckpt_metric <= ema_best_metric
                        ema_best_metric = min(val_ema_ckpt_metric, ema_best_metric)

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
                        k_best_checkpoints=keep_k_best_ckpts
                    )
                    logger.info('Checkpoints saved at: {}'.format(save_dir), print_line=True)

                if self.tb_log_writter is not None and self.is_master_node:
                    lr_list = self.scheduler.retrieve_lr(self.optimizer)
                    for g_id, lr_val in enumerate(lr_list):
                        self.tb_log_writter.add_scalar('LR/Group-{}'.format(g_id), round(lr_val, 6), epoch)
                    self.tb_log_writter.add_scalar('Train/Loss', round(train_loss, 2), epoch)
                    self.tb_log_writter.add_scalar('Val/Loss', round(val_loss, 2), epoch)
                    self.tb_log_writter.add_scalar('Common/Best Metric', round(self.best_metric, 2), epoch)
                    if val_ema_loss is not None:
                        self.tb_log_writter.add_scalar('Val_EMA/Loss', round(val_ema_loss, 2), epoch)

                    # If val checkpoint metric is different from loss, add that too
                    if self.ckpt_metric != 'loss':
                        self.tb_log_writter.add_scalar('Train/{}'.format(self.ckpt_metric.title()),
                                                       round(train_ckpt_metric, 2), epoch)
                        self.tb_log_writter.add_scalar('Val/{}'.format(self.ckpt_metric.title()),
                                                       round(val_ckpt_metric, 2), epoch)
                        if val_ema_ckpt_metric is not None:
                            self.tb_log_writter.add_scalar('Val_EMA/{}'.format(self.ckpt_metric.title()),
                                                           round(val_ema_ckpt_metric, 2), epoch)

                if self.max_iterations_reached and self.is_master_node:
                    logger.info('Max. iterations for training reached')
                    break

        except KeyboardInterrupt:
            if self.is_master_node:
                logger.log('Keyboard interruption. Exiting from early training')
        except Exception as e:
            if self.is_master_node:
                if 'out of memory' in str(e):
                    logger.log('OOM exception occured')
                    n_gpus = getattr(self.opts, "dev.num_gpus", 1)
                    for dev_id in range(n_gpus):
                        mem_summary = torch.cuda.memory_summary(device=torch.device('cuda:{}'.format(dev_id)),
                                                                abbreviated=True)
                        logger.log('Memory summary for device id: {}'.format(dev_id))
                        print(mem_summary)
                else:
                    logger.log('Exception occurred that interrupted the training. {}'.format(str(e)))
                    print(e)
                    raise e
        finally:
            use_distributed = getattr(self.opts, "ddp.use_distributed", False)
            if use_distributed:
                torch.distributed.destroy_process_group()

            torch.cuda.empty_cache()

            if self.is_master_node and self.tb_log_writter is not None:
                self.tb_log_writter.close()

            if self.is_master_node:
                train_end_time = time.time()
                hours, rem = divmod(train_end_time - train_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
                logger.log('Training took {}'.format(train_time_str))
            try:
                exit(0)
            except Exception as e:
                pass
            finally:
                pass

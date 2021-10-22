#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from metrics import Statistics, metric_monitor
from utils.ddp_utils import is_master
from torch.cuda.amp import autocast
from utils import logger
import time
from engine.utils import print_summary
from common import DEFAULT_LOG_FREQ


class Evaluator(object):
    def __init__(self, opts, model, eval_loader):
        super(Evaluator, self).__init__()

        self.opts = opts

        self.model = model

        self.eval_loader = eval_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.is_master_node = is_master(opts)

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)

        self.metric_names = getattr(opts, "stats.name", ['loss'])
        if isinstance(self.metric_names, str):
            self.metric_names = [self.metric_names]
        assert isinstance(self.metric_names, list), "Type of metric names should be list. Got: {}".format(
            type(self.metric_names))

        if 'loss' in self.metric_names:
            self.metric_names.pop(self.metric_names.index('loss'))

        self.ckpt_metric = getattr(self.opts, "stats.checkpoint_metric", "top1")
        assert self.ckpt_metric in self.metric_names, \
            "Checkpoint metric should be part of metric names. Metric names: {}, Checkpoint metric: {}".format(
                self.metric_names, self.ckpt_metric)

        if self.is_master_node:
            print_summary(opts=self.opts,  model=self.model)

    def eval_fn(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)
        device = getattr(self.opts, "dev.device", torch.device('cpu'))

        evaluation_stats = Statistics(metric_names=self.metric_names, is_master_node=self.is_master_node)

        model.eval()
        if model.training and self.is_master_node:
            logger.warning('Model is in training mode. Switching to evaluation mode')
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.eval_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.eval_loader):
                input_img, target_label = batch['image'], batch['label']

                # move data to device
                input_img = input_img.to(device)
                target_label = target_label.to(device)
                batch_size = input_img.shape[0]

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)

                processed_samples += batch_size
                metrics = metric_monitor(pred_label=pred_label, target_label=target_label, loss=0.0,
                                         use_distributed=self.use_distributed, metric_names=self.metric_names)

                evaluation_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(epoch=-1,
                                                  n_processed_samples=processed_samples,
                                                  total_samples=total_samples,
                                                  elapsed_time=epoch_start_time,
                                                  learning_rate=0.0
                                                  )

        evaluation_stats.epoch_summary(epoch=-1, stage="evaluation")

    def run(self):
        eval_start_time = time.time()
        self.eval_fn(model=self.model)
        eval_end_time = time.time() - eval_start_time
        logger.log('Evaluation took {} seconds'.format(eval_end_time))
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import time
from torch.cuda.amp import autocast

from metrics import Statistics, metric_monitor
from utils.ddp_utils import is_master
from utils import logger
from utils.common_utils import move_to_device
from engine.utils import print_summary
from common import DEFAULT_LOG_FREQ, SUPPORTED_VIDEO_CLIP_VOTING_FN


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

        self.metric_names = getattr(opts, "stats.val", ["loss"])
        if isinstance(self.metric_names, str):
            self.metric_names = [self.metric_names]
        assert isinstance(
            self.metric_names, list
        ), "Type of metric names should be list. Got: {}".format(
            type(self.metric_names)
        )

        if "loss" in self.metric_names:
            self.metric_names.pop(self.metric_names.index("loss"))

        self.ckpt_metric = getattr(self.opts, "stats.checkpoint_metric", "top1")
        assert (
            self.ckpt_metric in self.metric_names
        ), "Checkpoint metric should be part of metric names. Metric names: {}, Checkpoint metric: {}".format(
            self.metric_names, self.ckpt_metric
        )

        if self.is_master_node:
            print_summary(opts=self.opts, model=self.model)

        # inference modality based eval function
        self.eval_fn = self.eval_fn_image
        inference_modality = getattr(opts, "common.inference_modality", "image")
        if inference_modality is not None and inference_modality.lower() == "video":
            self.eval_fn = self.eval_fn_video

    def eval_fn_image(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            metric_names=self.metric_names, is_master_node=self.is_master_node
        )

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.eval_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.eval_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                input_img, target_label = batch["image"], batch["label"]

                batch_size = input_img.shape[0]

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)

                processed_samples += batch_size
                metrics = metric_monitor(
                    self.opts,
                    pred_label=pred_label,
                    target_label=target_label,
                    loss=0.0,
                    use_distributed=self.use_distributed,
                    metric_names=self.metric_names,
                )

                evaluation_stats.update(
                    metric_vals=metrics, batch_time=0.0, n=batch_size
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=-1,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=-1, stage="evaluation")

    def eval_fn_video(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            metric_names=self.metric_names, is_master_node=self.is_master_node
        )

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        num_clips_per_video = getattr(self.opts, "sampler.bs.clips_per_video", 1)
        voting_fn = getattr(
            self.opts, "model.video_classification.clip_out_voting_fn", "sum"
        )
        if voting_fn is None:
            voting_fn = "sum"
        voting_fn = voting_fn.lower()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.eval_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.eval_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                input_img, target_label = batch["image"], batch["label"]
                # target_label is Batch*Num_clips
                batch_size_ = target_label.shape[0]
                batch_size = batch_size_ // num_clips_per_video
                if batch_size_ != (batch_size * num_clips_per_video):
                    logger.log(
                        "Skipping batch. Expected batch size= {}. Got: (bxc:{}x{})".format(
                            batch_size_, batch_size, num_clips_per_video
                        )
                    )
                    continue

                with autocast(enabled=self.mixed_precision_training):
                    # prediction
                    pred_label = model(input_img)

                target_label = target_label.reshape(batch_size, num_clips_per_video)
                # label is the same for all clips in the video
                target_label = target_label[:, 0]
                pred_label = pred_label.reshape(batch_size, num_clips_per_video, -1)

                if voting_fn == "sum":
                    pred_label = torch.sum(pred_label, dim=1)
                elif voting_fn == "max":
                    pred_label = torch.max(pred_label, dim=1)
                else:
                    logger.error(
                        "--model.video-classification.clip-out-fusion-fn can be {}. Got: {}".format(
                            SUPPORTED_VIDEO_CLIP_VOTING_FN, voting_fn
                        )
                    )

                processed_samples += batch_size
                metrics = metric_monitor(
                    self.opts,
                    pred_label=pred_label,
                    target_label=target_label,
                    loss=0.0,
                    use_distributed=self.use_distributed,
                    metric_names=self.metric_names,
                )

                evaluation_stats.update(
                    metric_vals=metrics, batch_time=0.0, n=batch_size
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=-1,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=-1, stage="evaluation")

    def run(self):
        eval_start_time = time.time()
        self.eval_fn(model=self.model)
        eval_end_time = time.time() - eval_start_time
        logger.log("Evaluation took {} seconds".format(eval_end_time))

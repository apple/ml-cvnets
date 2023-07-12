#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import time

import torch

from common import DEFAULT_LOG_FREQ, SUPPORTED_VIDEO_CLIP_VOTING_FN
from engine.utils import autocast_fn, get_batch_size, get_log_writers
from metrics.stats import Statistics
from options.parse_args import parse_validation_metric_names
from utils import logger
from utils.common_utils import move_to_device
from utils.ddp_utils import is_master


class Evaluator(object):
    # Note: "test_loader" used to be named "eval_loader". We recently renamed data-related "eval_*" names to "test_*"
    #   to follow the standard train/val/test terminology. Engine-related names (eval_engine, is_evaluation, evaluator,
    #   etc.) remained unchanged. One of the reasons was to prevent "eval_engine.py"->"test_engine.py" being
    #   recognized a test suite by pytest.
    def __init__(self, opts, model, test_loader):
        super(Evaluator, self).__init__()

        self.opts = opts

        self.model = model

        self.test_loader = test_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.is_master_node = is_master(opts)
        self.stage_name = getattr(opts, "common.eval_stage_name", "evaluation")

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        self.mixed_precision_dtype = getattr(
            opts, "common.mixed_precision_dtype", "float16"
        )

        (
            self.metric_names,
            self.ckpt_metric,
            self.ckpt_submetric,
        ) = parse_validation_metric_names(self.opts)

        self.log_writers = get_log_writers(self.opts, save_location=None)

        # inference modality based eval function
        self.eval_fn = self.eval_fn_image
        inference_modality = getattr(opts, "common.inference_modality", "image")
        if inference_modality is not None and inference_modality.lower() == "video":
            self.eval_fn = self.eval_fn_video

    def eval_fn_image(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            opts=self.opts,
            metric_names=self.metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()
        if model.training and self.is_master_node:
            logger.warning("Model is in training mode. Switching to evaluation mode")
            model.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.test_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.test_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]

                batch_size = get_batch_size(samples)

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)

                processed_samples += batch_size

                evaluation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={
                        "loss": torch.tensor(0.0, dtype=torch.float, device=self.device)
                    },
                    batch_time=0.0,
                    batch_size=batch_size,
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=0,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=0, stage=self.stage_name)

    def eval_fn_video(self, model):
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            opts=self.opts,
            metric_names=self.metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
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
            total_samples = len(self.test_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.test_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]
                # target_label is Batch*Num_clips
                batch_size_ = get_batch_size(samples)
                batch_size = batch_size_ // num_clips_per_video
                if batch_size_ != (batch_size * num_clips_per_video):
                    logger.log(
                        "Skipping batch. Expected batch size= {}. Got: (bxc:{}x{})".format(
                            batch_size_, batch_size, num_clips_per_video
                        )
                    )
                    continue

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)

                targets = targets.reshape(batch_size, num_clips_per_video)
                # label is the same for all clips in the video
                targets = targets[:, 0]
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

                evaluation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={
                        "loss": torch.tensor(0.0, dtype=torch.float, device=self.device)
                    },
                    batch_time=0.0,
                    batch_size=batch_size,
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=0,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=0, stage=self.stage_name)

    def run(self):
        eval_start_time = time.time()
        self.eval_fn(model=self.model)
        eval_end_time = time.time() - eval_start_time
        logger.log("Evaluation took {} seconds".format(eval_end_time))

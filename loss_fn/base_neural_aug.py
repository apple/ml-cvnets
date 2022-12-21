#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
import argparse
import math
from torch.nn import functional as F

from utils import logger
from utils.ddp_utils import is_master

from . import BaseCriteria


class BaseNeuralAug(BaseCriteria):
    __supported_metrics = ["psnr"]

    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)

        perceptual_metric = getattr(opts, "loss.neural_aug.perceptual_metric", "psnr")
        is_master_node = is_master(opts)
        if perceptual_metric is None and is_master_node:
            logger.error(
                "Perceptual metric can't be none. "
                "Please specify perceptual metric using --loss.neural-aug.perceptual-metric argument"
            )
        if not isinstance(perceptual_metric, str) and is_master_node:
            logger.error(
                "The type of perceptual metric is not string. Got: {}".format(
                    type(perceptual_metric)
                )
            )
        perceptual_metric = perceptual_metric.lower()
        target_value = getattr(opts, "loss.neural_aug.target_value", None)

        self.curriculumn_learning = False
        self.iteration_based_training = getattr(
            opts, "scheduler.is_iteration_based", False
        )
        self.target_str = f"{target_value}"
        if perceptual_metric == "psnr":
            if target_value is None and is_master_node:
                logger.error("Target PSNR value can not be None.")

            if isinstance(target_value, (int, float)):
                if target_value < 0:
                    if is_master_node:
                        logger.error(
                            "PSNR value should be >= 0 in {}. Got: {}".format(
                                self.__class__.__name__, target_value
                            )
                        )
                # compute target MSE using below equation
                # # PSNR = 20 log10(255) - 10 log10(MSE)
                target_mse = 10.0 ** ((20.0 * math.log10(255.0) - target_value) / 10.0)
                self.target_value = torch.ones(size=(1,), dtype=torch.float).fill_(
                    target_mse
                )
                self.target_str = f"{target_value}"
            elif isinstance(target_value, (list, tuple)) and len(target_value) == 2:
                start_target_value = target_value[0]
                end_target_value = target_value[1]

                if start_target_value < 0 or end_target_value < 0:
                    if is_master_node:
                        logger.error(
                            "PSNR value should be >= 0 in {}. Got: {}".format(
                                self.__class__.__name__, target_value
                            )
                        )

                # compute target MSE using below equation
                # # PSNR = 20 log10(255) - 10 log10(MSE)
                start_target_mse = 10.0 ** (
                    (20.0 * math.log10(255.0) - start_target_value) / 10.0
                )
                end_target_mse = 10.0 ** (
                    (20.0 * math.log10(255.0) - end_target_value) / 10.0
                )

                max_steps = (
                    getattr(opts, "scheduler.max_iterations", None)
                    if self.iteration_based_training
                    else getattr(opts, "scheduler.max_epochs", None)
                )

                if max_steps is None and is_master_node:
                    logger.error(
                        "Please specify {}. Got None.".format(
                            "--scheduler.max-iterations"
                            if self.iteration_based_training
                            else "--scheduler.max-epochs"
                        )
                    )

                curriculum_method = getattr(
                    opts, "loss.neural_aug.curriculum_method", None
                )
                if curriculum_method in CURRICULUMN_METHOD.keys():
                    self.target_value = CURRICULUMN_METHOD[curriculum_method](
                        start=start_target_mse, end=end_target_mse, period=max_steps
                    )
                else:
                    raise NotImplementedError

                self.curriculumn_learning = True
                self.target_str = f"[{start_target_value}, {end_target_value}]"
            else:
                raise NotImplementedError

            # the maximum possible MSE error is computed as:
            # a = torch.ones((3, H, W)) * 255.0 # Max. input value is 255.0
            # b = torch.zeros((3, H, W)) # min. input value is 0.0
            # mse = torch.mean( (a -b) ** 2)

            self.alpha = 100.0 / 65025.0  # 65025 is the maximum mse
        else:
            if is_master_node:
                logger.error(
                    "Supported perceptual metrics are: {}. Got: {}".format(
                        self.__supported_metrics, perceptual_metric
                    )
                )
        self.perceptual_metric = perceptual_metric

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def _forward_psnr(
        self, input_tensor: Tensor, augmented_tensor: Tensor, *args, **kwargs
    ) -> Tensor:
        squared_err = ((augmented_tensor - input_tensor) * 255.0) ** 2
        # [B, C, H, W] --> [B]
        pred_mse = torch.mean(squared_err, dim=[1, 2, 3])

        # compute L1 loss between target and current MSE
        if self.curriculumn_learning:
            step = (
                kwargs.get("iterations", 0)
                if self.iteration_based_training
                else kwargs.get("epoch", 0)
            )
            if step >= len(self.target_value):
                step = -1
            target_mse = self.target_value[step]
        else:
            target_mse = self.target_value

        loss_na = F.smooth_l1_loss(
            input=pred_mse,
            target=target_mse.expand_as(pred_mse).to(
                device=pred_mse.device, dtype=pred_mse.dtype
            ),
            reduction="mean",
        )

        loss_na = loss_na * self.alpha
        return loss_na

    def forward_neural_aug(
        self, input_tensor: Tensor, augmented_tensor: Tensor, *args, **kwargs
    ) -> Tensor:

        if self.perceptual_metric == "psnr":
            loss_na = self._forward_psnr(
                input_tensor=input_tensor,
                augmented_tensor=augmented_tensor,
                *args,
                **kwargs,
            )
            return loss_na
        else:
            logger.error(
                "Supported perceptual metrics are {}. Got: {}".format(
                    self.__supported_metrics, self.perceptual_metric
                )
            )

    def repr_na(self):
        return (
            "\n\ttarget_metric={}"
            "\n\ttarget_value={}"
            "\n\tcurriculum_learning={}".format(
                self.perceptual_metric,
                self.target_str,
                self.curriculumn_learning,
            )
        )

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def linear_curriculumn(start, end, period):
    """This function implements linear curriculumn"""
    return torch.linspace(start=start, end=end, steps=period + 1, dtype=torch.float)


def cosine_curriculumn(start, end, period):
    """This function implements cosine curriculumn"""

    curr = [
        end + 0.5 * (start - end) * (1 + math.cos(math.pi * i / (period + 1)))
        for i in range(period + 1)
    ]

    curr = torch.tensor(curr, dtype=torch.float)
    return curr


CURRICULUMN_METHOD = {
    "linear": linear_curriculumn,
    "cosine": cosine_curriculumn,
}

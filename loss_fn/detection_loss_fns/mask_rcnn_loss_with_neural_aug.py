#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import Tensor
from typing import Dict, List
import argparse

from utils import logger

from . import register_detection_loss_fn

from .mask_rcnn_loss import MaskRCNNLoss
from ..base_neural_aug import BaseNeuralAug


@register_detection_loss_fn(name="mask_rcnn_loss_with_na")
class MaskRCNNLossWithNA(MaskRCNNLoss, BaseNeuralAug):
    """Mask RCNN loss with neural augmentation"""

    def __init__(self, opts, *args, **kwargs):
        MaskRCNNLoss.__init__(self, opts, *args, **kwargs)
        BaseNeuralAug.__init__(self, opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self,
        input_sample: Dict[str, List],
        prediction: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if not isinstance(prediction, Dict):
            logger.error(
                "Prediction needs to be an instance of Dict and must contain logits and augmented_tensor"
                " as keys"
            )

        augmented_tensor = prediction.pop("augmented_tensor", None)

        if augmented_tensor is None:
            loss = MaskRCNNLoss.forward(
                self, input_sample=input_sample, prediction=prediction, *args, **kwargs
            )
            return loss

        if not isinstance(input_sample, Dict):
            logger.error(
                "Input is expected as a Dictionary containing atleast image as a key"
            )

        if not {"image"}.issubset(input_sample.keys()):
            logger.error(
                "Input is expected as a Dictionary containing atleast image as a key. Got: {}".format(
                    input_sample.keys()
                )
            )

        input_image_sample = input_sample["image"]
        if isinstance(input_image_sample, List):
            # if its a list of images, stack them
            input_image_sample = torch.stack(input_image_sample, dim=0)

        loss_na = self.forward_neural_aug(
            input_tensor=input_image_sample,
            augmented_tensor=augmented_tensor,
            *args,
            **kwargs,
        )

        loss = MaskRCNNLoss.forward(
            self, input_sample=input_sample, prediction=prediction, *args, **kwargs
        )

        loss["total_loss"] += loss_na
        loss["na_loss"] = loss_na
        return loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{self.extra_repr() + self.repr_na()}\n)".replace(
            "\n\n", "\n"
        )

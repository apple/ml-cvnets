#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from torch import Tensor
from typing import Dict
import argparse

from utils import logger

from . import register_classification_loss_fn

from .cross_entropy import ClsCrossEntropy
from ..base_neural_aug import BaseNeuralAug


@register_classification_loss_fn(name="cross_entropy_with_na")
class CrossEntropyWithNA(ClsCrossEntropy, BaseNeuralAug):
    """Cross entropy with Perceptual loss for classification tasks with neural augmentation"""

    def __init__(self, opts, *args, **kwargs):
        ClsCrossEntropy.__init__(self, opts, *args, **kwargs)
        BaseNeuralAug.__init__(self, opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self, input_sample: Tensor, prediction: Dict, target: Tensor, *args, **kwargs
    ) -> Dict[str, Tensor]:
        if not isinstance(prediction, Dict):
            logger.error(
                "Prediction needs to be an instance of Dict and must contain logits and augmented_tensor"
                " as keys"
            )

        if not {"augmented_tensor", "logits"}.issubset(prediction.keys()):
            logger.error(
                "Prediction needs to be an instance of Dict and must contain logits and augmented_tensor"
                " as keys. Got keys: {}".format(prediction.keys())
            )

        augmented_tensor = prediction.get("augmented_tensor", None)
        logits = prediction.get("logits", None)

        if augmented_tensor is None:
            ce_loss = ClsCrossEntropy.forward(
                self,
                input_sample=input_sample,
                prediction=logits,
                target=target,
                *args,
                **kwargs
            )
            return {"total_loss": ce_loss}

        loss_na = self.forward_neural_aug(
            input_tensor=input_sample,
            augmented_tensor=augmented_tensor,
            *args,
            **kwargs
        )

        ce_loss = ClsCrossEntropy.forward(
            self,
            input_sample=augmented_tensor,
            prediction=logits,
            target=target,
            *args,
            **kwargs
        )

        return {
            "total_loss": loss_na + ce_loss,
            "na_loss": loss_na,
            "cls_loss": ce_loss,
        }

    def __repr__(self):
        repr_str = (
            "{}(\n\tignore_idx={}"
            "\n\tclass_wts={}"
            "\n\tlabel_smoothing={}{}"
            "\n)".format(
                self.__class__.__name__,
                self.ignore_idx,
                self.use_class_wts,
                self.label_smoothing,
                self.repr_na(),
            )
        )
        return repr_str

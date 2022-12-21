#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Union, Dict
import argparse

from utils import logger

from . import register_segmentation_loss_fn
from .cross_entropy import SegCrossEntropy

from ..base_neural_aug import BaseNeuralAug


@register_segmentation_loss_fn(name="seg_cross_entropy_with_na")
class SegCrossEntropyWithNA(SegCrossEntropy, BaseNeuralAug):
    """Cross entropy with Perceptual loss for segmentation tasks with neural augmentation"""

    def __init__(self, opts, *args, **kwargs):
        SegCrossEntropy.__init__(self, opts, *args, **kwargs)
        BaseNeuralAug.__init__(self, opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self,
        input_sample: Tensor,
        prediction: Union[Dict, Tensor, Tuple[Tensor, Tensor]],
        target: Tensor,
        *args,
        **kwargs
    ) -> Dict:

        if isinstance(prediction, (Tuple, Tensor)):
            seg_loss = super().forward(
                input_sample=input_sample,
                prediction=prediction,
                target=target,
                *args,
                **kwargs
            )
            return {"total_loss": seg_loss}
        elif isinstance(prediction, Dict):
            if not {"augmented_tensor", "segmentation_output"}.issubset(
                prediction.keys()
            ):
                logger.error(
                    "Prediction needs to be an instance of Dict and must contain segmentation_output and augmented_tensor"
                    " as keys. Got keys: {}".format(prediction.keys())
                )
            augmented_tensor = prediction.get("augmented_tensor", None)
            segmentation_output = prediction.get("segmentation_output", None)
            if augmented_tensor is None:
                seg_loss = SegCrossEntropy.forward(
                    self,
                    input_sample=input_sample,
                    prediction=segmentation_output,
                    target=target,
                    *args,
                    **kwargs
                )
                return {"total_loss": seg_loss}

            seg_loss = SegCrossEntropy.forward(
                self,
                input_sample=input_sample,
                prediction=segmentation_output,
                target=target,
                *args,
                **kwargs
            )

            loss_na = self.forward_neural_aug(
                input_tensor=input_sample,
                augmented_tensor=augmented_tensor,
                *args,
                **kwargs
            )

            return {
                "total_loss": loss_na + seg_loss,
                "na_loss": loss_na,
                "seg_loss": seg_loss,
            }
        else:
            raise NotImplementedError

    def __repr__(self):
        repr_str = (
            "{}(\n\tweighted_loss={}\n\tignore_idx={}\n\tlabel_smoothing={}{}".format(
                self.__class__.__name__,
                self.weighted_loss,
                self.ignore_idx,
                self.label_smoothing,
                self.repr_na(),
            )
        )

        if self.aux_wt > 0:
            repr_str += "\n\taux_wt={}".format(self.aux_wt)
        return repr_str + "\n)"

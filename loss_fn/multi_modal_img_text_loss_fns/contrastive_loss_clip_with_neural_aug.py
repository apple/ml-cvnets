#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict, Any
import argparse

from . import register_multi_modal_img_text_loss_fns
from .contrastive_loss_clip import ContrastiveLossClip

from ..base_neural_aug import BaseNeuralAug


@register_multi_modal_img_text_loss_fns(name="contrastive_loss_clip_with_na")
class ContrastiveLossClipWithNA(ContrastiveLossClip, BaseNeuralAug):
    """CLIP Loss function for multi-modal image-text training with neural augmentation"""

    def __init__(self, opts, *args, **kwargs):
        ContrastiveLossClip.__init__(self, opts, *args, **kwargs)
        BaseNeuralAug.__init__(self, opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self, input_sample: Dict, prediction: Dict, target: Any, *args, **kwargs
    ) -> Dict:

        augmented_tensor = prediction.get("augmented_tensor")
        if augmented_tensor is None:
            return ContrastiveLossClip.forward(
                self,
                input_sample=input_sample,
                prediction=prediction,
                target=target,
                *args,
                **kwargs
            )
        elif "augmented_tensor" in prediction and "image" in input_sample:
            contrastive_loss = ContrastiveLossClip.forward(
                self,
                input_sample=input_sample,
                prediction=prediction,
                target=target,
                *args,
                **kwargs
            )

            loss_na = self.forward_neural_aug(
                input_tensor=input_sample["image"],
                augmented_tensor=augmented_tensor,
                *args,
                **kwargs
            )
            contrastive_loss["total_loss"] = (
                contrastive_loss.pop("total_loss") + loss_na
            )
            contrastive_loss["na_loss"] = loss_na
            return contrastive_loss
        else:
            raise NotImplementedError

    def __repr__(self):
        return "{}({}\n)".format(self.__class__.__name__, self.repr_na())

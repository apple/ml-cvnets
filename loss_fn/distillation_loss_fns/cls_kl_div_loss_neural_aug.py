#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Union, Dict
import argparse

from utils import logger

from . import register_distillation_loss_fn
from .cls_kl_div_loss import ClsKLDivLoss

from ..base_neural_aug import BaseNeuralAug


@register_distillation_loss_fn(name="cls_kl_div_loss_with_na")
class ClsKLDivLossWithNA(ClsKLDivLoss, BaseNeuralAug):
    """
    KLDiv loss with Perceptual loss for distillation
    """

    def __init__(self, opts, *args, **kwargs):
        BaseNeuralAug.__init__(self, opts, *args, **kwargs)
        ClsKLDivLoss.__init__(self, opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def extra_repr(self) -> str:
        return super().extra_repr() + self.repr_na()

    def forward(
        self,
        input_sample: Tensor,
        prediction: Union[Dict, Tensor],
        target: Tensor,
        *args,
        **kwargs
    ) -> Dict:

        if isinstance(prediction, Tensor):
            kl_loss = super().forward(
                input_sample=input_sample,
                prediction=prediction,
                target=target,
                *args,
                **kwargs
            )
            return {"total_loss": kl_loss}
        elif isinstance(prediction, Dict):
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
                kl_loss = ClsKLDivLoss.forward(
                    self,
                    input_sample=input_sample,
                    prediction=logits,
                    target=target,
                    *args,
                    **kwargs
                )
                return {"total_loss": kl_loss}

            kl_loss = ClsKLDivLoss.forward(
                self,
                input_sample=augmented_tensor,
                prediction=logits,
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
                "total_loss": loss_na + kl_loss,
                "na_loss": loss_na,
                "kl_loss": kl_loss,
            }
        else:
            raise NotImplementedError

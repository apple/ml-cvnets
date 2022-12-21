#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, Tuple, Dict

import torch
from torch import Tensor
from torch.nn import functional as F

from utils.tensor_utils import gather_all_features
from . import BaseCriteria, register_multi_modal_img_text_loss_fns


@register_multi_modal_img_text_loss_fns(name="contrastive_loss_clip")
class ContrastiveLossClip(BaseCriteria):
    """CLIP Loss function for multi-modal image-text training"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.rank = getattr(opts, "ddp.rank", 0)
        self.use_distributed = getattr(opts, "ddp.use_distributed", False)
        self.device = getattr(opts, "dev.device", torch.device("cpu"))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self, input_sample: Any, prediction: Dict, target: Any, *args, **kwargs
    ) -> Dict[str, Tensor]:

        image_features = prediction.pop("image", None)
        text_features = prediction.pop("text", None)

        if image_features is None or text_features is None:
            # if either image features or text features is None, then loss can't be computed.
            # simply return 0.0
            return {
                "total_loss": torch.tensor(0.0, dtype=torch.float, device=self.device),
            }

        assert image_features is not None
        assert text_features is not None

        logit_scale = prediction.pop("logit_scale", 1.0)

        # we need to aggregate
        gathered_image_features, gathered_text_features = gather_features(
            image_features=image_features,
            text_features=text_features,
            use_distributed=self.use_distributed,
        )
        # compute logits
        # [B, d] x [BW x d]^T --> [B, BW]
        logits_per_image = logit_scale * (
            image_features @ gathered_text_features.transpose(0, 1)
        )
        # [B, d] x [BW, d]^T --> [B, BW]
        logits_per_text = logit_scale * (
            text_features @ gathered_image_features.transpose(0, 1)
        )

        # generate labels
        num_logits = logits_per_image.shape[0]
        contrastive_labels = torch.arange(
            num_logits, device=logits_per_image.device, dtype=torch.long
        )

        # shift the labels by rank id
        contrastive_labels = contrastive_labels + (num_logits * self.rank)

        text_loss = F.cross_entropy(logits_per_text, contrastive_labels) * 0.5
        image_loss = F.cross_entropy(logits_per_image, contrastive_labels) * 0.5
        total_loss = image_loss + text_loss
        return {
            "total_loss": total_loss,
            "image_loss": image_loss,
            "text_loss": text_loss,
            "logit_scale": logit_scale,
        }

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


def gather_features(
    image_features: Tensor, text_features: Tensor, use_distributed: bool
) -> Tuple[Tensor, Tensor]:
    """
    Helper function that allows us to gather image and text features from all DDP ranks in a differentiable manner
    """
    if use_distributed:
        # gather features from all ranks
        # [B, d] x W --> [BW, d] where W is the world size
        gathered_image_features = gather_all_features(features=image_features, dim=0)
        # [B, d] x W --> [BW, d] where W is the world size
        gathered_text_features = gather_all_features(features=text_features, dim=0)
        return gathered_image_features, gathered_text_features
    return image_features, text_features

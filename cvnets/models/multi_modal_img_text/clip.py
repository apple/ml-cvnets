#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from cvnets.image_projection_layers import build_image_projection_head
from cvnets.image_projection_layers.base_image_projection import (
    get_in_feature_dimension,
)
from cvnets.models import MODEL_REGISTRY, BaseAnyNNModel, get_model
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from cvnets.models.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImageText,
)
from cvnets.text_encoders import BaseTextEncoder, build_text_encoder
from utils import logger


@MODEL_REGISTRY.register(name="clip", type="multi_modal_image_text")
class CLIP(BaseMultiModalImageText):
    """Base class for multi-modal image-text data"""

    def __init__(
        self,
        opts: argparse.Namespace,
        image_encoder: BaseImageEncoder,
        text_encoder: BaseTextEncoder,
        *args,
        **kwargs
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.image_encoder: BaseImageEncoder = image_encoder
        self.text_encoder: BaseTextEncoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.use_distributed = getattr(opts, "ddp.use_distributed", False)
        self.cache_text_features_zero_shot = getattr(
            opts,
            "model.multi_modal_image_text.clip.cache_text_features_zero_shot",
            False,
        )
        self.cached_text_features = None
        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.multi-modal-image-text.clip.projection-dim",
            type=int,
            default=256,
            help="Project image and text features to this dimensionality",
        )
        group.add_argument(
            "--model.multi-modal-image-text.clip.cache-text-features-zero-shot",
            action="store_true",
            help="Cache text features for zero-shot during inference",
        )

        return parser

    def reset_parameters(self) -> None:
        """Reset weights image and text models"""
        torch.nn.init.constant_(self.logit_scale, math.log(1.0 / 0.07))

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        image_param_list, image_lr_mult = self.image_encoder.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name="image_encoder.",
            *args,
            **kwargs
        )
        # The learning rate list from image encoder returns 1.0 as a LR multiplier.
        # Update the learning rate to the specified value.
        image_lr_mult = [self.lr_multiplier_img_encoder] * len(image_lr_mult)

        text_param_list, text_lr_mult = self.text_encoder.get_trainable_parameters(
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name="text_encoder.",
            *args,
            **kwargs
        )
        # The learning rate list from text encoder returns 1.0 as a LR multiplier.
        # Update the learning rate to the specified value.
        text_lr_mult = [self.lr_multiplier_text_encoder] * len(text_lr_mult)

        # We need to add the logit scale
        logit_scale_param_list = [
            {
                "params": [self.logit_scale],
                "weight_decay": 0.0,
                "param_names": ["logit_scale"],
            }
        ]
        logit_scale_lr_mult = [1.0] * len(logit_scale_param_list)

        return (
            image_param_list + text_param_list + logit_scale_param_list,
            image_lr_mult + text_lr_mult + logit_scale_lr_mult,
        )

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        height = 224
        width = 224
        vocab_size = 10
        seq_length = 5
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )

        text_tensor = torch.randint(
            low=0, high=vocab_size, size=(batch_size, seq_length)
        ).long()

        return {
            "samples": {"image": img_tensor, "text": text_tensor},
            "targets": text_tensor,
        }

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def forward(self, input: Dict, *args, **kwargs) -> Dict:

        images = input.get("image", None)
        text_tokens = input.get("text", None)
        padding_mask = input.get("padding_mask", None)

        # [B, C, H, W] --> [B, d]
        # where B is the batch size, C is number of image channels, H is height and W is Width
        image_encoder_out = self.image_encoder(images)
        augmented_tensor = None
        if isinstance(image_encoder_out, Dict):
            if not {"augmented_tensor", "logits"}.issubset(image_encoder_out.keys()):
                logger.error(
                    "Output of image classifier must contain logits and augmented_tensor"
                    " as keys. Got keys: {}".format(image_encoder_out.keys())
                )
            image_embeddings = image_encoder_out["logits"]
            augmented_tensor = image_encoder_out["augmented_tensor"]
        elif isinstance(image_encoder_out, Tensor):
            image_embeddings = image_encoder_out
        else:
            logger.error("The output of image encoder should be either Dict or Tensor")

        # [B, N] --> [B, d] (for single-caption per image)
        # or [B, Cl, M, N] --> [d, Cl] (for zero-shot)
        # where N in caption len, M is number of captions per image, and Cl is number of classes per image

        if self.cache_text_features_zero_shot and not self.training:
            # For zero-shot image classification, we can cache text features as they are the same for all images
            if self.cached_text_features is None:
                text_embeddings = self.text_encoder(
                    text_tokens=text_tokens, key_padding_mask=padding_mask
                )
                self.cached_text_features = text_embeddings
            text_embeddings = self.cached_text_features
        else:
            text_embeddings = self.text_encoder(
                text_tokens=text_tokens, key_padding_mask=padding_mask
            )

        if (
            not self.training
            and (
                text_embeddings.shape[0] == image_embeddings.shape[1]
            )  # d_text == d_image
            and (
                text_embeddings.shape[1] != image_embeddings.shape[0]
            )  # N_classes != Batch_image
        ):
            # This means that we are running a zero-shot set-up.
            # [B x d] x [d x N] --> [B, N]
            zero_shot_image_logits = 100.0 * image_embeddings @ text_embeddings
            return {
                "image": None,
                "text": None,
                "logit_scale": self._exponentiate_and_clip_logits(),
                "zero_shot_image_logits": zero_shot_image_logits,
                "augmented_tensor": None,
            }
        else:
            return {
                "image": image_embeddings,
                "text": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
                "zero_shot_image_logits": None,
                "augmented_tensor": augmented_tensor,
            }

    @classmethod
    def build_model(cls, opts, *args, **kwargs) -> BaseAnyNNModel:
        """Helper function to build the multi-modal image-text model"""
        projection_dim = getattr(
            opts, "model.multi_modal_image_text.clip.projection_dim", -1
        )
        if projection_dim < 1:
            logger.error("Projection dimension should be > 1. Got: {}")

        image_encoder: BaseImageEncoder = get_model(
            opts=opts, category="classification", *args, **kwargs
        )
        text_encoder: BaseTextEncoder = build_text_encoder(
            opts=opts, projection_dim=projection_dim, *args, **kwargs
        )

        # replace the classifier in image encoder with the task specific classifier
        image_encoder.classifier = update_image_classifier(
            opts,
            image_classifier=image_encoder.classifier,
            projection_dim=projection_dim,
        )

        model = cls(
            opts,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            *args,
            **kwargs
        )

        if getattr(opts, "model.multi_modal_image_text.freeze_batch_norm"):
            cls.freeze_norm_layers(opts, model)
        return model


def update_image_classifier(
    opts, image_classifier: nn.Module, projection_dim: int, *args, **kwargs
) -> nn.Module:
    in_features = get_in_feature_dimension(image_classifier)
    new_img_classifier = build_image_projection_head(
        opts, in_dim=in_features, out_dim=projection_dim
    )
    return new_img_classifier

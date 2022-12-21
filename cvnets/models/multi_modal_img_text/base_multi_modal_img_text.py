#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
import argparse
from typing import Optional, Tuple, Dict

from ...layers import norm_layers_tuple
from ...misc.init_utils import initialize_weights


class BaseMultiModalImageText(nn.Module):
    """Base class for multi-modal image-text data"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        self.lr_multiplier_img_encoder = getattr(
            opts, "model.multi_modal_image_text.lr_multiplier_img_encoder", 1.0
        )
        self.lr_multiplier_text_encoder = getattr(
            opts, "model.multi_modal_image_text.lr_multiplier_text_encoder", 1.0
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model specific arguments"""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.multi-modal-image-text.name",
            type=str,
            default=None,
            help="Name of the multi-modal image-text model",
        )

        group.add_argument(
            "--model.multi-modal-image-text.lr-multiplier-img-encoder",
            type=float,
            default=1.0,
            help="LR multiplier for the image encoder in {}".format(cls.__name__),
        )
        group.add_argument(
            "--model.multi-modal-image-text.lr-multiplier-text-encoder",
            type=float,
            default=1.0,
            help="LR multiplier for the text encoder in {}".format(cls.__name__),
        )

        group.add_argument(
            "--model.multi-modal-image-text.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone",
        )
        group.add_argument(
            "--model.multi-modal-image-text.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )

        return parser

    def reset_parameters(self) -> None:
        """Reset weights of a given layer"""
        initialize_weights(opts=self.opts, modules=self.modules())

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        raise NotImplementedError

    def profile_model(self, input: Tensor) -> Optional[Tuple[Tensor, float, float]]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def freeze_norm_layers(self) -> None:
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        raise NotImplementedError

    def forward(self, input: Dict, *args, **kwargs) -> Dict:
        raise NotImplementedError

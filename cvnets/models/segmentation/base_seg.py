#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import argparse
from typing import Optional, Tuple, Dict

from ..classification import BaseEncoder
from ... import parameter_list
from ...layers import norm_layers_tuple
from ...misc.init_utils import initialize_weights


class BaseSegmentation(nn.Module):
    """Base class for segmentation networks"""

    def __init__(self, opts, encoder: BaseEncoder, *args, **kwargs) -> None:
        super().__init__()
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier", 1.0)
        assert isinstance(
            encoder, BaseEncoder
        ), "encoder should be an instance of BaseEncoder"
        self.encoder: BaseEncoder = encoder

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add segmentation model specific arguments"""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--model.segmentation.name", type=str, default=None, help="Model name"
        )
        group.add_argument(
            "--model.segmentation.n-classes",
            type=int,
            default=20,
            help="Number of classes in the dataset",
        )
        group.add_argument(
            "--model.segmentation.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained segmentation model. Useful for evaluation",
        )
        group.add_argument(
            "--model.segmentation.lr-multiplier",
            type=float,
            default=1.0,
            help="Multiply the learning rate in segmentation network (e.g., decoder)",
        )
        group.add_argument(
            "--model.segmentation.classifier-dropout",
            type=float,
            default=0.1,
            help="Dropout rate in classifier",
        )
        group.add_argument(
            "--model.segmentation.use-aux-head",
            action="store_true",
            help="Use auxiliary output",
        )
        group.add_argument(
            "--model.segmentation.aux-dropout",
            default=0.1,
            type=float,
            help="Dropout in auxiliary branch",
        )

        group.add_argument(
            "--model.segmentation.output-stride",
            type=int,
            default=None,
            help="Output stride in classification network",
        )
        group.add_argument(
            "--model.segmentation.replace-stride-with-dilation",
            action="store_true",
            help="Replace stride with dilation",
        )

        group.add_argument(
            "--model.segmentation.activation.name",
            default=None,
            type=str,
            help="Non-linear function type",
        )
        group.add_argument(
            "--model.segmentation.activation.inplace",
            action="store_true",
            help="Inplace non-linear functions",
        )
        group.add_argument(
            "--model.segmentation.activation.neg-slope",
            default=0.1,
            type=float,
            help="Negative slope in leaky relu",
        )
        group.add_argument(
            "--model.segmentation.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )

        group.add_argument(
            "--model.segmentation.use-level5-exp",
            action="store_true",
            help="Use output of Level 5 expansion layer in base feature extractor",
        )

        group.add_argument(
            "--model.segmentation.finetune-pretrained-model",
            action="store_true",
            help="Finetune a pretrained model",
        )
        group.add_argument(
            "--model.segmentation.n-pretrained-classes",
            type=int,
            default=None,
            help="Number of pre-trained classes",
        )
        return parser

    @staticmethod
    def reset_layer_parameters(layer, opts) -> None:
        """Reset weights of a given layer"""
        initialize_weights(opts=opts, modules=layer.modules())

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

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
        img_channels = 3
        height = 224
        width = 224
        n_classes = 10
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        label_tensor = torch.randint(
            low=0, high=n_classes, size=(batch_size, height, width)
        ).long()
        return {"samples": img_tensor, "targets": label_tensor}

    def update_classifier(self, opts, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        raise NotImplementedError

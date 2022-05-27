#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
import argparse
from typing import Optional, Tuple


from ..classification import BaseEncoder
from ... import parameter_list
from ...layers import norm_layers_tuple
from ...misc.init_utils import initialize_weights


class BaseSegmentation(nn.Module):
    """Base class for segmentation networks"""

    def __init__(self, opts, encoder: BaseEncoder) -> None:
        super(BaseSegmentation, self).__init__()
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier", 1.0)
        assert isinstance(
            encoder, BaseEncoder
        ), "encoder should be an instance of BaseEncoder"
        self.encoder: BaseEncoder = encoder

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add segmentation model specific arguments"""
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

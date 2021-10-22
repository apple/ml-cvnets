#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
import argparse

from cvnets.layers import norm_layers_tuple

from ..classification import BaseEncoder
from ... import parameter_list
from ...misc.init_utils import initialize_weights


class BaseSegmentation(nn.Module):
    def __init__(self, opts, encoder: BaseEncoder):
        super(BaseSegmentation, self).__init__()
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier", 1.0)
        assert isinstance(encoder, BaseEncoder), "encoder should be an instance of BaseEncoder"
        self.encoder: BaseEncoder = encoder

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    @staticmethod
    def reset_layer_parameters(layer, opts):
        # weight initialization
        initialize_weights(opts=opts, modules=layer.modules())

    def get_trainable_parameters(self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False):
        param_list = parameter_list(named_parameters=self.named_parameters,
                                    weight_decay=weight_decay,
                                    no_decay_bn_filter_bias=no_decay_bn_filter_bias)
        return param_list, [1.0] * len(param_list)

    def profile_model(self, input: Tensor):
        raise NotImplementedError

    def freeze_norm_layers(self):
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

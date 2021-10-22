#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional
from utils.math_utils import make_divisible

from ..layers import AdaptiveAvgPool2d, ConvLayer, get_activation_fn, LinearLayer
from ..modules import BaseModule
from ..misc.profiler import module_profile


class SqueezeExcitation(BaseModule):
    """
        This class defines the SE module
            https://arxiv.org/abs/1709.01507
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 squeeze_factor: Optional[int] = 4,
                 scale_fn_name: Optional[str] = 'sigmoid'
                 ) -> None:
        squeeze_channels = max(make_divisible(in_channels // squeeze_factor, 8), 32)

        fc1 = ConvLayer(opts=opts, in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1,
                        bias=True, use_norm=False, use_act=True)
        fc2 = ConvLayer(opts=opts, in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1,
                        bias=True, use_norm=False, use_act=False)
        if scale_fn_name == "sigmoid":
            act_fn = get_activation_fn(act_type="sigmoid")
        elif scale_fn_name == "hard_sigmoid":
            act_fn = get_activation_fn(act_type="hard_sigmoid", inplace=True)
        else:
            raise NotImplementedError

        super(SqueezeExcitation, self).__init__()
        self.se_layer = nn.Sequential()
        self.se_layer.add_module(name="global_pool", module=AdaptiveAvgPool2d(output_size=1))
        self.se_layer.add_module(name="fc1", module=fc1)
        self.se_layer.add_module(name="fc2", module=fc2)
        self.se_layer.add_module(name="scale_act", module=act_fn)

        self.in_channels = in_channels
        self.squeeze_factor = squeeze_factor
        self.scale_fn = scale_fn_name

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se_layer(x)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        _, params, macs = module_profile(module=self.se_layer, x=input)
        return input, params, macs

    def __repr__(self):
        return '{}(in_channels={}, squeeze_factor={}, scale_fn={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.squeeze_factor,
            self.scale_fn
        )

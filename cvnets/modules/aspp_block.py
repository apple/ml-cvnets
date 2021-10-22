#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
import numpy as np

from ..layers import BaseLayer, ConvLayer, AdaptiveAvgPool2d, SeparableConv, Dropout, NormActLayer
from ..modules import BaseModule
from ..misc.profiler import module_profile


class ASPP(BaseModule):
    """
        ASPP module defined in DeepLab papers:
            https://arxiv.org/abs/1606.00915
            https://arxiv.org/abs/1706.05587
    """
    def __init__(self, opts,
                 in_channels: int,
                 out_channels: int,
                 atrous_rates: Tuple,
                 is_sep_conv: Optional[bool] = False,
                 dropout: Optional[float] = 0.1,
                 *args, **kwargs
                 ):
        in_proj = ConvLayer(opts=opts, in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, use_norm=True, use_act=True)
        out_proj = ConvLayer(opts=opts, in_channels=5 * out_channels, out_channels=out_channels,
                             kernel_size=1, stride=1, use_norm=True, use_act=True)
        aspp_layer = ASPPSeparableConv if is_sep_conv else ASPPConv

        assert len(atrous_rates) == 3

        modules = [in_proj]
        modules.extend(
            [
                aspp_layer(opts=opts, in_channels=in_channels, out_channels=out_channels, dilation=rate) for rate in atrous_rates
            ]
        )
        modules.append(ASPPPooling(opts=opts, in_channels=in_channels, out_channels=out_channels))

        super(ASPP, self).__init__()
        self.convs = nn.ModuleList(modules)
        self.project = out_proj

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        self.is_sep_conv = is_sep_conv
        self.n_atrous_branches = len(atrous_rates)
        self.dropout = Dropout(p=dropout) if 0.0 < dropout < 1.0 else None

    def forward(self, x):
        out = []
        for conv in self.convs:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        out = self.project(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params, macs = 0.0, 0.0
        res = []
        for c in self.convs:
            out, p, m = module_profile(module=c, x=input)
            params += p
            macs += m
            res.append(out)
        res = torch.cat(res, dim=1)

        out, p, m = module_profile(module=self.project, x=res)
        params += p
        macs += m
        return out, params, macs

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, atrous_rates={}, is_aspp_sep={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.atrous_rates,
            self.is_sep_conv,
        )


class ASPPConv(ConvLayer):
    def __init__(self, opts, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, use_norm=True,
            use_act=True, dilation=dilation
        )

    def adjust_atrous_rate(self, rate):
        self.block.conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.block.conv.padding = rate


class ASPPSeparableConv(SeparableConv):
    def __init__(self, opts, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPSeparableConv, self).__init__(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation
        )

    def adjust_atrous_rate(self, rate):
        self.dw_conv.block.conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.dw_conv.block.conv.padding = rate


class ASPPPooling(BaseLayer):
    def __init__(self, opts, in_channels: int, out_channels: int) -> None:

        super(ASPPPooling, self).__init__()
        self.aspp_pool = nn.Sequential()
        self.aspp_pool.add_module(
            name="global_pool",
            module=AdaptiveAvgPool2d(output_size=1)
        )
        self.aspp_pool.add_module(
            name="conv_1x1",
            module=ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, use_norm=True, use_act=True
            )
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.shape[-2:]
        x = self.aspp_pool(x)
        x = F.interpolate(x, size=x_size, mode="bilinear", align_corners=False)
        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        out, params, macs = module_profile(module=self.aspp_pool, x=input)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)
        return out, params, macs

    def __repr__(self):
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels
        )

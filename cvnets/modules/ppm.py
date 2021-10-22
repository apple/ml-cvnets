#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from ..layers import ConvLayer, AdaptiveAvgPool2d, Dropout2d
from ..modules import BaseModule
from ..misc.profiler import module_profile


class PPM(BaseModule):
    """
        PSPNet module as define in the PSPNet paper:
            https://arxiv.org/abs/1612.01105
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 out_channels: int,
                 pool_sizes: Optional[Tuple] = (1, 2, 3, 6),
                 dropout: Optional[float] = 0.1
                 ) -> None:
        projection_dim = max(int(in_channels / len(pool_sizes)), 64)
        psp_branches = []
        for ps in pool_sizes:
            cbr_layer = ConvLayer(opts=opts, in_channels=in_channels, out_channels=projection_dim,
                                  kernel_size=1, stride=1, use_norm=True, use_act=True)
            branch = nn.Sequential()
            branch.add_module(name="pool_".format(ps), module=AdaptiveAvgPool2d(output_size=ps))
            branch.add_module(name="conv_1x1", module=cbr_layer)
            psp_branches.append(branch)

        channels_after_concat = in_channels + (projection_dim * len(pool_sizes))
        conv_3x3 = ConvLayer(opts=opts, in_channels=channels_after_concat, out_channels=out_channels,
                             kernel_size=3, stride=1, use_norm=True, use_act=True)
        super(PPM, self).__init__()
        self.psp_branches = nn.ModuleList(psp_branches)
        self.dropout = Dropout2d(p=dropout) if 0.0 < dropout < 1.0 else None
        self.fusion = conv_3x3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes
        self.inner_channels = projection_dim

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.size()
        res = [x]
        for psp_branch in self.psp_branches:
            out = psp_branch(x)
            out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
            res.append(out)
        res = torch.cat(res, dim=1)
        if self.dropout is not None:
            res = self.dropout(res)
        return self.fusion(res)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params, macs = 0.0, 0.0
        res = [input]
        input_size = input.size()
        for psp_branch in self.psp_branches:
            out, p, m = module_profile(module=psp_branch, x=input)
            out = F.interpolate(out, input_size[2:], mode='bilinear', align_corners=True)
            params += p
            macs += m
            res.append(out)
        res = torch.cat(res, dim=1)

        res, p, m = module_profile(module=self.fusion, x=res)
        return res, params + p, macs + m

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, pool_sizes={}, inner_channels={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.pool_sizes,
            self.inner_channels
        )
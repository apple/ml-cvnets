#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List
import torch.nn.functional as F

from ..layers import ConvLayer, UpSample
from ..modules import BaseModule
from ..misc.profiler import module_profile


class FPModule(BaseModule):
    """
        Inspired from the PSP module in the PSPNet paper:
            https://arxiv.org/abs/1612.01105
        Difference: Replaces the average pooling with Upsample function
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 out_channels: int,
                 scales: Optional[Tuple or List] = (0.25, 0.5, 2.0),
                 ) -> None:
        projection_dim = max(int(in_channels / len(scales)), 32)
        fp_branches = []
        for scale in scales:
            cbr_layer = ConvLayer(opts=opts, in_channels=in_channels, out_channels=projection_dim,
                                  kernel_size=1, stride=1, use_norm=True, use_act=True)
            branch = nn.Sequential()
            branch.add_module(
                name="scale_".format(scale),
                module=UpSample(scale_factor=scale, mode="bilinear", align_corners=False)
            )
            branch.add_module(name="conv_1x1", module=cbr_layer)
            fp_branches.append(branch)

        channels_after_concat = in_channels + (projection_dim * len(scales))
        conv_3x3 = ConvLayer(opts=opts, in_channels=channels_after_concat, out_channels=out_channels,
                             kernel_size=3, stride=1, use_norm=True, use_act=True)
        super(FPModule, self).__init__()
        self.fp_branches = nn.ModuleList(fp_branches)
        self.fusion = conv_3x3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.size()
        res = [x]
        for psp_branch in self.fp_branches:
            out = psp_branch(x)
            out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
            res.append(out)
        return self.fusion(torch.cat(res, dim=1))

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        params, macs = 0.0, 0.0
        res = [input]
        input_size = input.size()
        for psp_branch in self.fp_branches:
            out, p, m = module_profile(module=psp_branch, x=input)
            out = F.interpolate(out, input_size[2:], mode='bilinear', align_corners=True)
            params += p
            macs += m
            res.append(out)
        res = torch.cat(res, dim=1)

        res, p, m = module_profile(module=self.fusion, x=res)
        return res, params + p, macs + m

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, scales={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.scales
        )
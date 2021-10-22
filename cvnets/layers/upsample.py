#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor


class UpSample(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpSample, self).__init__(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

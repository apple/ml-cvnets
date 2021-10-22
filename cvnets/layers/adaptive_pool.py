#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple = 1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

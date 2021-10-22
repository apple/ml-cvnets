#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return '{}(kernel_size={}, stride={})'.format(self.__class__.__name__, self.kernel_size, self.stride)


class AvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size: tuple, stride: Optional[tuple] = None, padding: Optional[tuple] = (0, 0),
                 ceil_mode: Optional[bool] = False, count_include_pad: Optional[bool] = True, divisor_override: Optional[bool] = None):
        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return '{}(upscale_factor={})'.format(self.__class__.__name__, self.upscale_factor)
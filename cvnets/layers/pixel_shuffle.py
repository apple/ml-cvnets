#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor


class PixelShuffle(nn.PixelShuffle):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__(upscale_factor=upscale_factor)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return '{}(upscale_factor={})'.format(self.__class__.__name__, self.upscale_factor)

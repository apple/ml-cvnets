#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor


class Flatten(nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        """
            Flattens a contiguous range of dims into a tensor.
            :param start_dim: first dim to flatten (default = 1).
            :param end_dim: last dim to flatten (default = -1).
        """
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

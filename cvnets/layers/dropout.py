#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor


class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples \
        from a Bernoulli distribution.

        :param p: probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0


class Dropout2d(nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples \
        from a Bernoulli distribution.

        :param p: probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        super(Dropout2d, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0
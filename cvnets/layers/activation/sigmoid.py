#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="sigmoid")
class Sigmoid(nn.Sigmoid):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        return input, 0.0, 0.0

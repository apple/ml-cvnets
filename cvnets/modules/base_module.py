#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Tuple


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Tensor or Tuple[Tensor]) -> Tensor or Tuple[Tensor]:
        raise NotImplementedError

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import Tensor
from .base_layer import BaseLayer
import random
from utils.math_utils import bound_fn
from collections import OrderedDict


class RandomApply(BaseLayer):
    """
        Apply layers randomly during training
    """
    def __init__(self, module_list: list, keep_p: float = 0.8):
        super(RandomApply, self).__init__()
        self._modules = OrderedDict()
        for idx, module in enumerate(module_list):
            self._modules[str(idx)] = module

        self.module_indexes = [i for i in range(1, len(self._modules))]
        n_blocks = len(self.module_indexes)
        k = int(round(n_blocks * keep_p))
        self.keep_k = bound_fn(min_val=1, max_val=n_blocks, value=k)

    def forward(self, x):
        if self.training:
            indexes = [0] + sorted(random.sample(self.module_indexes, k=self.keep_k))
            for idx in indexes:
                x = self._modules[str(idx)](x)
        else:
            for idx, layer in self._modules.items():
                x = layer(x)
        return x

    def profile_module(self, x, *args, **kwargs) -> (Tensor, float, float):
        params, macs = 0.0, 0.0
        for idx, layer in self._modules.items():
            x, p, m = layer.profile_module(x)
            params += p
            macs += m
        return x, params, macs

    def __repr__(self):
        format_string = self.__class__.__name__ + ' (apply_k (N={})={}, '.format(len(self._modules), self.keep_k)
        for idx, layer in self._modules.items():
            format_string += '\n\t {}'.format(layer)
        format_string += '\n)'
        return format_string
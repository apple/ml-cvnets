#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import random
from typing import List, Optional, Tuple

from utils.math_utils import bound_fn

from .base_layer import BaseLayer


class RandomApply(BaseLayer):
    """
    This layer randomly applies a list of modules during training.

    Args:
        module_list (List): List of modules
        keep_p (Optional[float]): Keep P modules from the list during training. Default: 0.8 (or 80%)
    """

    def __init__(
        self, module_list: List, keep_p: Optional[float] = 0.8, *args, **kwargs
    ) -> None:
        super().__init__()
        n_modules = len(module_list)
        self.module_list = module_list

        self.module_indexes = [i for i in range(1, n_modules)]
        k = int(round(n_modules * keep_p))
        self.keep_k = bound_fn(min_val=1, max_val=n_modules, value=k)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            indexes = [0] + sorted(random.sample(self.module_indexes, k=self.keep_k))
            for idx in indexes:
                x = self.module_list[idx](x)
        else:
            for layer in self.module_list:
                x = layer(x)
        return x

    def profile_module(self, x, *args, **kwargs) -> Tuple[Tensor, float, float]:
        params, macs = 0.0, 0.0
        for layer in self.module_list:
            x, p, m = layer.profile_module(x)
            params += p
            macs += m
        return x, params, macs

    def __repr__(self):
        format_string = "{}(apply_k (N={})={}, ".format(
            self.__class__.__name__, len(self.module_list), self.keep_k
        )
        for layer in self.module_list:
            format_string += "\n\t {}".format(layer)
        format_string += "\n)"
        return format_string

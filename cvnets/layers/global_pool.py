#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor
import argparse

from utils import logger

from .base_layer import BaseLayer

pool_types = ['mean', 'rms', 'abs']


class GlobalPool(BaseLayer):
    def __init__(self, pool_type='mean', keep_dim=False):
        """
            Global pooling
            :param pool_type: Global pool operation type (mean, rms, abs)
            :param keep_dim: Keep dimensions the same as the input or not
        """
        super(GlobalPool, self).__init__()
        if pool_type not in pool_types:
            logger.error('Supported pool types are: {}. Got {}'.format(pool_types, pool_type))
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument('--model.layer.global-pool', type=str, default='mean', help='Which global pooling?')
        return parser

    def _global_pool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.pool_type == 'rms':
            x = x ** 2
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == 'abs':
            x = torch.mean(torch.abs(x), dim=[-2, -1], keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._global_pool(x)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

    def __repr__(self):
        return '{}(type={})'.format(self.__class__.__name__, self.pool_type)

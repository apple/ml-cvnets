#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional
import argparse

from utils import logger

from .base_layer import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: Optional[bool] = True,
                 *args, **kwargs) -> None:
        """
            Applies a linear transformation to the input data

            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--model.layer.linear-init', type=str, default='xavier_uniform',
                            help='Init type for linear layers')
        parser.add_argument('--model.layer.linear-init-std-dev', type=float, default=0.01,
                            help='Std deviation for Linear layers')
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, bias={})".format(
            self.__class__.__name__, self.in_features,
            self.out_features,
            True if self.bias is not None else False
        )
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs


class GroupLinear(BaseLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_groups: int,
                 bias: Optional[bool] = True,
                 feature_shuffle: Optional[bool] = False,
                 *args, **kwargs):
        """
            Applies a group linear transformation as defined in the following papers:
                https://arxiv.org/abs/1808.09029
                https://arxiv.org/abs/1911.12385
                https://arxiv.org/abs/2008.00623

            :param in_features: size of each input sample
            :param out_features: size of each output sample
            :param n_groups: Number of groups
            :param bias: Add bias (learnable) or not
            :param feature_shuffle: Mix output of each group after group linear transformation
            :param is_ws: Standardize weights or not (experimental)
        """
        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            logger.error(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            logger.error(err_msg)

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        super(GroupLinear, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.out_features = out_features
        self.in_features = in_features
        self.n_groups = n_groups
        self.feature_shuffle = feature_shuffle

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--model.layer.group-linear-init', type=str, default='xavier_uniform',
                            help='Init type for GLT layers')
        parser.add_argument('--model.layer.group-linear-init-std-dev', type=float, default=0.01,
                            help='Std deviation for GLT layers')
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def _forward(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor of shape [B, N] where B is batch size and N is the number of input features
        :return:
            Tensor of shape [B, M] where M is the number of output features
        """

        bsz = x.shape[0]
        # [B, N] -->  [B, g, N/g]
        x = x.reshape(bsz, self.n_groups, -1)

        # [B, g, N/g] --> [g, B, N/g]
        x = x.transpose(0, 1)
        # [g, B, N/g] x [g, N/g, M/g] --> [g, B, M/g]
        x = torch.bmm(x, self.weight)

        if self.bias is not None:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g, B, M/g] --> [B, M/g, g]
            x = x.permute(1, 2, 0)
            # [B, M/g, g] --> [B, g, M/g]
            x = x.reshape(bsz, self.n_groups, -1)
        else:
            # [g, B, M/g] --> [B, g, M/g]
            x = x.transpose(0, 1)

        return x.reshape(bsz, -1)

    def _glt_transform(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self._forward(x)
            return x
        elif x.dim() == 3:
            dim_0, dim_1, inp_dim = x.size()
            x = x.reshape(dim_1 * dim_0, -1)
            x = self._forward(x)
            x = x.reshape(dim_0, dim_1, -1)
            return x
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self._glt_transform(x)

    def __repr__(self):
        repr_str = '{}(in_features={}, out_features={}, groups={}, bias={}, shuffle={})'.format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.n_groups,
            True if self.bias is not None else False,
            self.feature_shuffle
        )
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):

        params = sum([p.numel() for p in self.parameters()])
        macs = params

        out_size = list(input.shape)
        out_size[-1] = self.out_features

        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs

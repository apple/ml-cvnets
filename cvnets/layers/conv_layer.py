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
from .normalization_layers import get_normalization_layer
from .non_linear_layers import get_activation_fn


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple or int, stride: tuple or int,
                 padding: tuple or int, dilation: int or tuple, groups: int, bias: bool, padding_mode: str,
                 *args, **kwargs):
        super(Conv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                     padding_mode=padding_mode)


class ConvLayer(BaseLayer):
    def __init__(self, opts, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True,
                 *args, **kwargs
                 ) -> None:
        """
            Applies a 2D convolution over an input signal composed of several input planes.
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: Add bias or not
            :param padding_mode: Padding mode. Default is zeros
            :param use_norm: Use normalization layer after convolution layer or not. Default is True.
            :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
            normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            logger.error('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            logger.error('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()

        conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                            padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        cls_name = "{} arguments".format(cls.__name__)
        group = parser.add_argument_group(title=cls_name, description=cls_name)
        group.add_argument('--model.layer.conv-init', type=str, default='kaiming_normal',
                           help='Init type for conv layers')
        parser.add_argument('--model.layer.conv-init-std-dev', type=float, default=None,
                            help='Std deviation for conv layers')
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ', normalization={}'.format(self.norm_name)

        if self.act_name is not None:
            repr_str += ', activation={}'.format(self.act_name)
        repr_str += ', bias={})'.format(self.bias)
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        if input.dim() != 4:
            logger.error(
                'Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}'.format(input.size()))

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, '{}!={}'.format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h // stride_h
        out_w = in_w // stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(size=(b, self.out_channels, out_h, out_w), dtype=input.dtype, device=input.device)
        # print(macs)
        return output, params, macs


class TransposeConvLayer(BaseLayer):
    def __init__(self, opts, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True,
                 padding: Optional[int or tuple] = (0, 0),
                 auto_padding: Optional[bool] = True):
        """
        Applies a 2D Transpose Convolution over an input signal composed of several input planes.
        :param opts: over an input signal composed of several input planes.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: move the kernel by this amount during convolution operation
        :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
        :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
        :param bias: Add bias or not
        :param padding_mode: Padding mode. Default is zeros
        :param use_norm: Use normalization layer after convolution layer or not. Default is True.
        :param use_act: Use activation layer after convolution layer/convolution layer followed by batch normalization
                        or not. Default is True.
        :param padding: Padding
        :param auto_padding: Compute padding automatically
        """
        super(TransposeConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, (tuple, list)):
            dilation = dilation[0]

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, int)

        if auto_padding:
            padding = (int((kernel_size[0] - 1)) * dilation, int((kernel_size[1] - 1)) * dilation)

        if in_channels % groups != 0:
            logger.error('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            logger.error('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()
        conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                        padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "relu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ', normalization={}'.format(self.norm_name)

        if self.act_name is not None:
            repr_str += ', activation={}'.format(self.act_name)
        repr_str += ')'
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        if input.dim() != 4:
            logger.error(
                'Conv2d requires 4-dimensional input (BxCxHxW). Provided input has shape: {}'.format(input.size()))

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, '{}!={}'.format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h * stride_h
        out_w = in_w * stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(size=(b, self.out_channels, out_h, out_w), dtype=input.dtype, device=input.device)
        # print(macs)
        return output, params, macs


class NormActLayer(BaseLayer):
    def __init__(self, opts, num_features):
        """
            Applies a normalization layer followed by activation layer over an input tensor
        :param opts: arguments
        :param num_features: number of feature planes in the input tensor
        """
        super(NormActLayer, self).__init__()

        block = nn.Sequential()

        self.norm_name = None
        norm_layer = get_normalization_layer(opts=opts, num_features=num_features)
        block.add_module(name="norm", module=norm_layer)
        self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type,
                                      inplace=inplace,
                                      negative_slope=neg_slope,
                                      num_parameters=num_features)
        block.add_module(name="act", module=act_layer)
        self.act_name = act_layer.__class__.__name__

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        # compute parameters
        params = sum([p.numel() for p in self.parameters()])
        macs = 0.0
        return input, params, macs

    def __repr__(self):
        repr_str = '{}(normalization={}, activation={})'.format(self.__class__.__name__, self.norm_type, self.act_type)
        return repr_str

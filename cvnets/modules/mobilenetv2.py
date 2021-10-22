#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from cvnets.modules import BaseModule
from torch import nn, Tensor
from utils.math_utils import make_divisible
from typing import Optional, Union

from ..misc.profiler import module_profile
from ..modules import SqueezeExcitation
from ..layers import ConvLayer, get_activation_fn


class InvertedResidualSE(BaseModule):
    """
        Inverted residual block w/ SE (MobileNetv3): https://arxiv.org/abs/1905.02244
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: Union[int, float],
                 use_hs: Optional[bool] = False,
                 dilation: Optional[int] = 1,
                 stride: Optional[int] = 1,
                 use_se: Optional[bool] = False
                 ) -> None:
        super(InvertedResidualSE, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if use_hs:
            act_fn = get_activation_fn(act_type="hard_swish", inplace= True)
        else:
            act_fn = get_activation_fn(act_type="relu", inplace=True)

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvLayer(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=False, use_norm=True))
            block.add_module(name="act_fn_1", module=act_fn)

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=False, use_norm=True, dilation=dilation)
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if use_se:
            se = SqueezeExcitation(opts=opts, in_channels=hidden_dim, squeeze_factor=4, scale_fn_name="hard_sigmoid")
            block.add_module(name="se", module=se)

        block.add_module(name="red_1x1",
                         module=ConvLayer(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_hs = use_hs
        self.use_se = use_se

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, use_hs={}, use_se={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_hs,
            self.use_se
        )


class InvertedResidual(BaseModule):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: Union[int, float],
                 dilation: int = 1
                 ) -> None:
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1",
                             module=ConvLayer(opts, in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                                              use_act=True, use_norm=True))

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(opts, in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3,
                             groups=hidden_dim, use_act=True, use_norm=True, dilation=dilation)
        )

        block.add_module(name="red_1x1",
                         module=ConvLayer(opts, in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                                          use_act=False, use_norm=True))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp, self.dilation
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Union, Tuple

from utils.math_utils import make_divisible

from . import BaseModule, SqueezeExcitation
from ..misc.profiler import module_profile
from ..layers import ConvLayer, get_activation_fn


class InvertedResidualSE(BaseModule):
    """
    This class implements the inverted residual block with squeeze-excitation unit, as described in
    `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        use_hs (Optional[bool]): Use hard sigmoid activation over relu. Default: False
        dilation (Optional[int]): Use conv with dilation. Default: 1
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        use_se (Optional[bool]): Use squeeze-excitation block. Default: False

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        expand_ratio: Union[int, float],
        use_hs: Optional[bool] = False,
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        if use_hs:
            act_fn = get_activation_fn(act_type="hard_swish", inplace=True)
        else:
            act_fn = get_activation_fn(act_type="relu", inplace=True)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=False,
                    use_norm=True,
                ),
            )
            block.add_module(name="act_fn_1", module=act_fn)

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=False,
                use_norm=True,
                dilation=dilation,
            ),
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if use_se:
            se = SqueezeExcitation(
                opts=opts,
                in_channels=hidden_dim,
                squeeze_factor=4,
                scale_fn_name="hard_sigmoid",
            )
            block.add_module(name="se", module=se)

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_hs = use_hs
        self.use_se = use_se
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, use_hs={}, use_se={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_hs,
            self.use_se,
        )


class InvertedResidual(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )

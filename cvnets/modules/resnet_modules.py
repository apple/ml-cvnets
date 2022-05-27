#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple

from ..layers import ConvLayer, Identity, get_activation_fn, Dropout
from ..modules import BaseModule
from ..misc.profiler import module_profile


class BasicResNetBlock(BaseModule):
    """
    This class defines the Basic block in the `ResNet model <https://arxiv.org/abs/1512.03385>`_
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        mid_channels (int): :math:`C_{mid}` from an expected tensor of size :math:`(N, C_{mid}, H_{out}, W_{out})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        stride (Optional[int]): Stride for convolution. Default: 1
        dilation (Optional[int]): Dilation for convolution. Default: 1
        dropout (Optional[float]): Dropout after second convolution. Default: 0.0

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    """

    expansion: int = 1

    def __init__(
        self,
        opts,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: Optional[int] = 1,
        dilation: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        cbr_1 = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            use_norm=True,
            use_act=True,
        )
        cb_2 = ConvLayer(
            opts=opts,
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_norm=True,
            use_act=False,
            dilation=dilation,
        )

        block = nn.Sequential()
        block.add_module(name="conv_batch_act_1", module=cbr_1)
        block.add_module(name="conv_batch_2", module=cb_2)
        if 0.0 < dropout < 1.0:
            block.add_module(name="dropout", module=Dropout(p=dropout))

        down_sample = Identity()
        if stride == 2:
            down_sample = ConvLayer(
                opts=opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                use_norm=True,
                use_act=False,
            )

        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.block = block
        self.down_sample = down_sample

        self.final_act = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=out_channels,
        )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.dropout = dropout

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        out = self.block(x)
        res = self.down_sample(x)
        out = out + res
        return self.final_act(out)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out, n_params, n_macs = module_profile(module=self.block, x=input)
        _, n_params_down, n_macs_down = module_profile(module=self.down_sample, x=input)
        return out, n_params + n_params_down, n_macs + n_macs_down

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, dilation={}, dropout={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.dilation,
            self.dropout,
        )


class BottleneckResNetBlock(BaseModule):
    """
    This class defines the Bottleneck block in the `ResNet model <https://arxiv.org/abs/1512.03385>`_
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        mid_channels (int): :math:`C_{mid}` from an expected tensor of size :math:`(N, C_{mid}, H_{out}, W_{out})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        stride (Optional[int]): Stride for convolution. Default: 1
        dilation (Optional[int]): Dilation for convolution. Default: 1
        dropout (Optional[float]): Dropout after third convolution. Default: 0.0

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    """

    expansion: int = 4

    def __init__(
        self,
        opts,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: Optional[int] = 1,
        dilation: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)

        cbr_1 = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )
        cbr_2 = ConvLayer(
            opts=opts,
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            use_norm=True,
            use_act=True,
            dilation=dilation,
        )
        cb_3 = ConvLayer(
            opts=opts,
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )
        block = nn.Sequential()
        block.add_module(name="conv_batch_act_1", module=cbr_1)
        block.add_module(name="conv_batch_act_2", module=cbr_2)
        block.add_module(name="conv_batch_3", module=cb_3)
        if 0.0 < dropout < 1.0:
            block.add_module(name="dropout", module=Dropout(p=dropout))

        down_sample = Identity()
        if stride == 2:
            down_sample = ConvLayer(
                opts=opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                use_norm=True,
                use_act=False,
            )
        elif in_channels != out_channels:
            down_sample = ConvLayer(
                opts=opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_norm=True,
                use_act=False,
            )

        super().__init__()
        self.block = block

        self.down_sample = down_sample
        self.final_act = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=out_channels,
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.dilation = dilation
        self.dropout = dropout

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        out = self.block(x)
        res = self.down_sample(x)
        out = out + res
        return self.final_act(out)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out, n_params, n_macs = module_profile(module=self.block, x=input)
        _, n_params_down, n_macs_down = module_profile(module=self.down_sample, x=input)
        return out, n_params + n_params_down, n_macs + n_macs_down

    def __repr__(self) -> str:
        return "{}(in_channels={}, mid_channels={}, out_channels={}, stride={}, dilation={}, dropout={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.mid_channels,
            self.out_channels,
            self.stride,
            self.dilation,
            self.dropout,
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import argparse
from typing import Optional, Dict, Tuple

from .base_seg_head import BaseSegHead
from . import register_segmentation_head
from ....layers import ConvLayer, UpSample, Dropout2d
from ....modules import ASPP
from ....misc.profiler import module_profile


@register_segmentation_head(name="deeplabv3")
class DeeplabV3(BaseSegHead):
    """
    This class defines the segmentation head in `DeepLabv3 architecture <https://arxiv.org/abs/1706.05587>`_
    Args:
        opts: command-line arguments
        enc_conf (Dict): Encoder input-output configuration at each spatial level
        use_l5_exp (Optional[bool]): Use features from expansion layer in Level5 in the encoder
    """

    def __init__(
        self, opts, enc_conf: Dict, use_l5_exp: Optional[bool] = False, *args, **kwargs
    ) -> None:
        atrous_rates = getattr(
            opts, "model.segmentation.deeplabv3.aspp_rates", (6, 12, 18)
        )
        out_channels = getattr(
            opts, "model.segmentation.deeplabv3.aspp_out_channels", 256
        )
        is_sep_conv = getattr(opts, "model.segmentation.deeplabv3.aspp_sep_conv", False)
        dropout = getattr(opts, "model.segmentation.deeplabv3.aspp_dropout", 0.1)

        super().__init__(opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)

        self.aspp = nn.Sequential()
        aspp_in_channels = (
            self.enc_l5_channels if not self.use_l5_exp else self.enc_l5_exp_channels
        )
        self.aspp.add_module(
            name="aspp_layer",
            module=ASPP(
                opts=opts,
                in_channels=aspp_in_channels,
                out_channels=out_channels,
                atrous_rates=atrous_rates,
                is_sep_conv=is_sep_conv,
                dropout=dropout,
            ),
        )

        self.classifier = ConvLayer(
            opts=opts,
            in_channels=out_channels,
            out_channels=self.n_seg_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )

        self.reset_head_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """DeepLabv3 specific arguments"""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.segmentation.deeplabv3.aspp-rates",
            type=tuple,
            default=(6, 12, 18),
            help="Atrous rates in DeepLabV3+ model",
        )
        group.add_argument(
            "--model.segmentation.deeplabv3.aspp-out-channels",
            type=int,
            default=256,
            help="Output channels of ASPP module",
        )
        group.add_argument(
            "--model.segmentation.deeplabv3.aspp-sep-conv",
            action="store_true",
            help="Separable conv in ASPP module",
        )
        group.add_argument(
            "--model.segmentation.deeplabv3.aspp-dropout",
            type=float,
            default=0.1,
            help="Dropout in ASPP module",
        )
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        # low resolution features
        x = enc_out["out_l5_exp"] if self.use_l5_exp else enc_out["out_l5"]
        # ASPP featues
        x = self.aspp(x)
        # classify
        x = self.classifier(x)
        return x

    def profile_module(self, enc_out: Dict) -> Tuple[Tensor, float, float]:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        params, macs = 0.0, 0.0

        if self.use_l5_exp:
            x, p, m = module_profile(module=self.aspp, x=enc_out["out_l5_exp"])
        else:
            x, p, m = module_profile(module=self.aspp, x=enc_out["out_l5"])
        params += p
        macs += m

        out, p, m = module_profile(module=self.classifier, x=x)
        params += p
        macs += m

        print(
            "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                self.__class__.__name__,
                "Params",
                round(params / 1e6, 3),
                "MACs",
                round(macs / 1e6, 3),
            )
        )
        return out, params, macs

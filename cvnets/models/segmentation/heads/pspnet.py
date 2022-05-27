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
from ....modules import PSP
from ....misc.profiler import module_profile


@register_segmentation_head(name="pspnet")
class PSPNet(BaseSegHead):
    """
    This class defines the segmentation head in `PSPNet architecture <https://arxiv.org/abs/1612.01105>`_
    Args:
        opts: command-line arguments
        enc_conf (Dict): Encoder input-output configuration at each spatial level
        use_l5_exp (Optional[bool]): Use features from expansion layer in Level5 in the encoder
    """

    def __init__(
        self, opts, enc_conf: dict, use_l5_exp: Optional[bool] = False, *args, **kwargs
    ) -> None:
        psp_out_channels = getattr(
            opts, "model.segmentation.pspnet.psp_out_channels", 512
        )
        psp_pool_sizes = getattr(
            opts, "model.segmentation.pspnet.psp_pool_sizes", [1, 2, 3, 6]
        )
        psp_dropout = getattr(opts, "model.segmentation.pspnet.psp_dropout", 0.1)

        super().__init__(opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)

        psp_in_channels = (
            self.enc_l5_channels if not self.use_l5_exp else self.enc_l5_exp_channels
        )
        self.psp_layer = PSP(
            opts=opts,
            in_channels=psp_in_channels,
            out_channels=psp_out_channels,
            pool_sizes=psp_pool_sizes,
            dropout=psp_dropout,
        )
        self.classifier = ConvLayer(
            opts=opts,
            in_channels=psp_out_channels,
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
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.segmentation.pspnet.psp-pool-sizes",
            type=int,
            nargs="+",
            default=[1, 2, 3, 6],
            help="Pool sizes in the PSPNet module",
        )
        group.add_argument(
            "--model.segmentation.pspnet.psp-out-channels",
            type=int,
            default=512,
            help="Output channels of PSPNet module",
        )
        group.add_argument(
            "--model.segmentation.pspnet.psp-dropout",
            type=float,
            default=0.1,
            help="Dropout in the PSPNet module",
        )
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        # low resolution features
        x = enc_out["out_l5_exp"] if self.use_l5_exp else enc_out["out_l5"]

        # Apply PSP layer
        x = self.psp_layer(x)

        out = self.classifier(x)

        return out

    def profile_module(self, enc_out: Dict) -> Tuple[Tensor, float, float]:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        params, macs = 0.0, 0.0

        if self.use_l5_exp:
            x, p, m = module_profile(module=self.psp_layer, x=enc_out["out_l5_exp"])
        else:
            x, p, m = module_profile(module=self.psp_layer, x=enc_out["out_l5"])
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

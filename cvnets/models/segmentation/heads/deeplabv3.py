#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional, Tuple

from torch import Tensor, nn

from cvnets.layers import ConvLayer2d
from cvnets.misc.init_utils import initialize_weights
from cvnets.models import MODEL_REGISTRY
from cvnets.models.segmentation.heads.base_seg_head import BaseSegHead
from cvnets.modules import ASPP
from options.parse_args import JsonValidator


@MODEL_REGISTRY.register(name="deeplabv3", type="segmentation_head")
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

        self.classifier = ConvLayer2d(
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

    def update_classifier(self, opts, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        in_channels = self.classifier.in_channels
        conv_layer = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        initialize_weights(opts, modules=conv_layer)
        self.classifier = conv_layer

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """DeepLabv3 specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.segmentation.deeplabv3.aspp-rates",
            type=JsonValidator(Tuple[int, int, int]),
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

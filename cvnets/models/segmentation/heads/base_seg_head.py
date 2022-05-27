#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Dict, Tuple
import argparse

from utils import logger

from ....misc.common import parameter_list
from ....misc.init_utils import initialize_weights
from ....layers import ConvLayer, Dropout2d, UpSample


class BaseSegHead(nn.Module):
    """
    Base class for segmentation heads
    """

    def __init__(self, opts, enc_conf: dict, use_l5_exp: Optional[bool] = False):
        enc_ch_l5_exp_out = _check_out_channels(enc_conf, "exp_before_cls")
        enc_ch_l5_out = _check_out_channels(enc_conf, "layer5")
        enc_ch_l4_out = _check_out_channels(enc_conf, "layer4")
        enc_ch_l3_out = _check_out_channels(enc_conf, "layer3")
        enc_ch_l2_out = _check_out_channels(enc_conf, "layer2")
        enc_ch_l1_out = _check_out_channels(enc_conf, "layer1")

        super().__init__()

        self.use_l5_exp = use_l5_exp
        self.enc_l5_exp_channels = enc_ch_l5_exp_out
        self.enc_l5_channels = enc_ch_l5_out
        self.enc_l4_channels = enc_ch_l4_out
        self.enc_l3_channels = enc_ch_l3_out
        self.enc_l2_channels = enc_ch_l2_out
        self.enc_l1_channels = enc_ch_l1_out

        self.n_seg_classes = getattr(opts, "model.segmentation.n_classes", 20)
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier", 1.0)
        self.classifier_dropout = getattr(
            opts, "model.segmentation.classifier_dropout", 0.1
        )
        self.output_stride = getattr(opts, "model.segmentation.output_stride", 16)

        self.aux_head = None
        if getattr(opts, "model.segmentation.use_aux_head", False):
            drop_aux = getattr(opts, "model.segmentation.aux_dropout", 0.1)
            inner_channels = max(int(self.enc_l4_channels // 4), 128)
            self.aux_head = nn.Sequential(
                ConvLayer(
                    opts=opts,
                    in_channels=self.enc_l4_channels,
                    out_channels=inner_channels,
                    kernel_size=3,
                    stride=1,
                    use_norm=True,
                    use_act=True,
                    bias=False,
                    groups=1,
                ),
                Dropout2d(drop_aux),
                ConvLayer(
                    opts=opts,
                    in_channels=inner_channels,
                    out_channels=self.n_seg_classes,
                    kernel_size=1,
                    stride=1,
                    use_norm=False,
                    use_act=False,
                    bias=True,
                    groups=1,
                ),
            )

        self.upsample_seg_out = None
        if self.output_stride != 1.0:
            self.upsample_seg_out = UpSample(
                scale_factor=self.output_stride, mode="bilinear", align_corners=True
            )

    def forward_aux_head(self, enc_out: Dict) -> Tensor:
        aux_out = self.aux_head(enc_out["out_l4"])
        return aux_out

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        raise NotImplementedError

    def forward(self, enc_out: Dict, *args, **kwargs) -> Tensor or Tuple[Tensor]:
        out = self.forward_seg_head(enc_out=enc_out)

        if self.upsample_seg_out is not None:
            # resize the mask based on given size
            mask_size = kwargs.get("orig_size", None)
            if mask_size is not None:
                self.upsample_seg_out.scale_factor = None
                self.upsample_seg_out.size = mask_size

            out = self.upsample_seg_out(out)

        if self.aux_head is not None and self.training:
            aux_out = self.forward_aux_head(enc_out=enc_out)
            return out, aux_out
        return out

    def reset_head_parameters(self, opts) -> None:
        # weight initialization
        initialize_weights(opts=opts, modules=self.modules())

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add segmentation head specific arguments"""
        group = parser.add_argument_group(
            title="Segmentation head arguments",
            description="Segmentation head arguments",
        )
        group.add_argument(
            "--model.segmentation.seg-head",
            type=str,
            default=None,
            help="Segmentation head",
        )

        return parser

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def get_trainable_parameters(
        self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False
    ):
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [self.lr_multiplier] * len(param_list)


def _check_out_channels(config: dict, layer_name: str) -> int:
    enc_ch_l: dict = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        logger.error(
            "Encoder does not define input-output mapping for {}: Got: {}".format(
                layer_name, config
            )
        )

    enc_ch_l_out = enc_ch_l.get("out", None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        logger.error(
            "Output channels are not defined in {} of the encoder. Got: {}".format(
                layer_name, enc_ch_l
            )
        )

    return enc_ch_l_out

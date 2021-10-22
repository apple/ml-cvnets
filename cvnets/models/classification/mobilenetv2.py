#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn
import argparse
from typing import Dict, List, Optional

from utils.math_utils import make_divisible

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.mobilenetv2 import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from ...modules import InvertedResidual


@register_cls_models("mobilenetv2")
class MobileNetV2(BaseEncoder):
    """
        This class implements the MobileNetv2 architecture
        Related paper: https://arxiv.org/abs/1801.04381
    """

    def __init__(self, opts, *args, **kwargs) -> None:

        width_mult = getattr(opts, "model.classification.mobilenetv2.width-multiplier", 1.0)
        num_classes = getattr(opts, "model.classification.n_classes", 1000)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` and `replace_stride_with_dilation` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        cfg = get_configuration(opts=opts)

        image_channels = 3
        input_channels = 32
        last_channel = 1280
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)
        super(MobileNetV2, self).__init__()
        last_channel = make_divisible(last_channel * max(1.0, width_mult), self.round_nearest)
        self.dilation = 1
        self.model_conf_dict = dict()

        self.conv_1 = ConvLayer(opts=opts, in_channels=image_channels, out_channels=input_channels,
                                kernel_size=3, stride=2, use_norm=True, use_act=True)
        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': input_channels}

        self.layer_1, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=cfg['layer1'],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer1'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_2, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=cfg['layer2'],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer2'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=cfg['layer3'],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest)
        self.model_conf_dict['layer3'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=[cfg['layer4'], cfg['layer4_a']],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest,
                                                      dilate=dilate_l4)
        self.model_conf_dict['layer4'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(opts=opts,
                                                      mv2_config=[cfg['layer5'], cfg['layer5_a']],
                                                      width_mult=width_mult,
                                                      input_channel=input_channels,
                                                      round_nearest=self.round_nearest,
                                                      dilate=dilate_l5)
        self.model_conf_dict['layer5'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = ConvLayer(opts=opts, in_channels=input_channels, out_channels=last_channel,
                                      kernel_size=1, stride=1, use_act=True, use_norm=True)
        self.model_conf_dict['exp_before_cls'] = {'in': input_channels, 'out': last_channel}

        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool",
            module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="classifier_dropout",
                module=Dropout(p=classifier_dropout)
            )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(in_features=last_channel, out_features=num_classes, bias=True)
        )

        self.model_conf_dict['cls'] = {'in': last_channel, 'out': num_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.classification.mobilenetv2.width-multiplier', type=float, default=1.0,
                           help="Width multiplier for MV2")

        return parser

    def _make_layer(self, opts, mv2_config: Dict or List, width_mult: float, input_channel: int,
                    round_nearest: Optional[int] = 8, dilate: Optional[bool] = False):
        prev_dilation = self.dilation
        mv2_block = nn.Sequential()
        count = 0

        if isinstance(mv2_config, Dict):
            mv2_config = [mv2_config]

        for cfg in mv2_config:
            t = cfg.get("expansion_ratio")
            c = cfg.get("out_channels")
            n = cfg.get("num_blocks")
            s = cfg.get("stride")

            output_channel = make_divisible(c * width_mult, round_nearest)

            for block_idx in range(n):
                stride = s if block_idx == 0 else 1
                block_name = "mv2_block_{}".format(count)
                if dilate and count == 0:
                    self.dilation *= stride
                    stride = 1

                layer = InvertedResidual(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=stride,
                    expand_ratio=t,
                    dilation=prev_dilation if count == 0 else self.dilation
                )
                mv2_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = output_channel
        return mv2_block, input_channel

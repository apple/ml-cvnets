#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn
import argparse
from typing import Tuple, Dict

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.resnet import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Identity, Dropout
from ...modules import BasicResNetBlock, BottleneckResNetBlock


@register_cls_models("resnet")
class ResNet(BaseEncoder):
    """
        This class implements the ResNet architecture with some modifications.
        Related paper: https://arxiv.org/pdf/1512.03385.pdf

        Modifications to the original ResNet architecture
        1. First 7x7 strided conv is replaced with 3x3 strided conv
        2. MaxPool operation is replaced with another 3x3 strided depth-wise conv
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        image_channels = 3
        input_channels = 64
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        cfg = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the ResNet backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(ResNet, self).__init__()
        self.dilation = 1
        self.model_conf_dict = dict()

        self.conv_1 = ConvLayer(opts=opts, in_channels=image_channels, out_channels=input_channels,
                                kernel_size=3, stride=2, use_norm=True, use_act=True)
        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': input_channels}

        self.layer_1 = ConvLayer(opts=opts, in_channels=input_channels, out_channels=input_channels,
                                 kernel_size=3, stride=2, use_norm=True, use_act=True, groups=input_channels)
        self.model_conf_dict['layer1'] = {'in': input_channels, 'out': input_channels}

        self.layer_2, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer2"]
                                                      )
        self.model_conf_dict['layer2'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer3"]
                                                      )
        self.model_conf_dict['layer3'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer4"],
                                                      dilate=dilate_l4
                                                      )
        self.model_conf_dict['layer4'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer5"],
                                                      dilate=dilate_l5
                                                      )
        self.model_conf_dict['layer5'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = Identity()
        self.model_conf_dict['exp_before_cls'] = {'in': input_channels, 'out': input_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="classifier_dropout", module=Dropout(p=classifier_dropout))
        self.classifier.add_module(name="classifier_fc",
                                   module=LinearLayer(in_features=input_channels, out_features=num_classes, bias=True))

        self.model_conf_dict['cls'] = {'in': input_channels, 'out': num_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def _make_layer(self, opts, in_channels: int, layer_config: Dict,
                    dilate: bool = False, *args, **kwargs) -> Tuple[nn.Sequential, int]:
        block_type = BottleneckResNetBlock if layer_config.get("block_type", "bottleneck").lower() == "bottleneck" \
            else BasicResNetBlock
        mid_channels = layer_config.get("mid_channels")
        num_blocks = layer_config.get("num_blocks", 2)
        stride = layer_config.get("stride", 1)

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        out_channels = block_type.expansion * mid_channels

        block = nn.Sequential()
        block.add_module(
            name="block_0",
            module=block_type(opts=opts, in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels,
                              stride=stride, dilation=previous_dilation)
        )

        for block_idx in range(1, num_blocks):
            block.add_module(
                name="block_{}".format(block_idx),
                module=block_type(opts=opts, in_channels=out_channels, mid_channels=mid_channels,
                                  out_channels=out_channels,
                                  stride=1, dilation=self.dilation)
            )

        return block, out_channels

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.classification.resnet.depth', type=int, default=50)
        return parser

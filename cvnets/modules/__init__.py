#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
# isort: skip_file
from cvnets.modules.base_module import BaseModule
from cvnets.modules.squeeze_excitation import SqueezeExcitation
from cvnets.modules.mobilenetv2 import InvertedResidual, InvertedResidualSE
from cvnets.modules.resnet_modules import (
    BasicResNetBlock,
    BottleneckResNetBlock,
)
from cvnets.modules.aspp_block import ASPP
from cvnets.modules.transformer import TransformerEncoder
from cvnets.modules.windowed_transformer import WindowedTransformerEncoder
from cvnets.modules.pspnet_module import PSP
from cvnets.modules.mobilevit_block import MobileViTBlock, MobileViTBlockv2
from cvnets.modules.feature_pyramid import FeaturePyramidNetwork
from cvnets.modules.ssd_heads import SSDHead, SSDInstanceHead
from cvnets.modules.efficientnet import EfficientNetBlock
from cvnets.modules.mobileone_block import MobileOneBlock, RepLKBlock
from cvnets.modules.swin_transformer_block import (
    SwinTransformerBlock,
    PatchMerging,
    Permute,
)
from cvnets.modules.regnet_modules import XRegNetBlock, AnyRegNetStage


__all__ = [
    "InvertedResidual",
    "InvertedResidualSE",
    "BasicResNetBlock",
    "BottleneckResNetBlock",
    "ASPP",
    "TransformerEncoder",
    "WindowedTransformerEncoder",
    "SqueezeExcitation",
    "PSP",
    "MobileViTBlock",
    "MobileViTBlockv2",
    "MobileOneBlock",
    "RepLKBlock",
    "FeaturePyramidNetwork",
    "SSDHead",
    "SSDInstanceHead",
    "EfficientNetBlock",
    "SwinTransformerBlock",
    "PatchMerging",
    "Permute",
    "XRegNetBlock",
    "AnyRegNetStage",
]

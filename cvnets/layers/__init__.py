#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import os
import importlib, inspect

from .base_layer import BaseLayer
from .conv_layer import ConvLayer, NormActLayer, TransposeConvLayer
from .sep_conv_layer import SeparableConv
from .linear_layer import LinearLayer, GroupLinear
from .global_pool import GlobalPool
from .identity import Identity
from .non_linear_layers import get_activation_fn
from .normalization_layers import get_normalization_layer, norm_layers_tuple
from .pixel_shuffle import PixelShuffle
from .upsample import UpSample
from .pooling import MaxPool2d, AvgPool2d
from .positional_encoding import PositionalEncoding
from .normalization_layers import AdjustBatchNormMomentum
from .adaptive_pool import AdaptiveAvgPool2d
from .flatten import Flatten
from .multi_head_attention import MultiHeadAttention
from .dropout import Dropout, Dropout2d

__all__ = [
    'ConvLayer',
    'SeparableConv',
    'NormActLayer',
    'TransposeConvLayer',
    'LinearLayer',
    'GroupLinear',
    'GlobalPool',
    'Identity',
    'PixelShuffle',
    'UpSample',
    'MaxPool2d',
    'AvgPool2d',
    'Dropout',
    'Dropout2d',
    'PositionalEncoding',
    'AdjustBatchNormMomentum',
    'Flatten',
    'MultiHeadAttention'
]


# iterate through all classes and fetch layer specific arguments
def layer_specific_args(parser: argparse.ArgumentParser):
    layer_dir = os.path.dirname(__file__)
    parsed_layers = []
    for file in os.listdir(layer_dir):
        path = os.path.join(layer_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            layer_name = file[: file.find(".py")] if file.endswith(".py") else file
            module = importlib.import_module("cvnets.layers." + layer_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseLayer) and name not in parsed_layers:
                    parser = cls.add_arguments(parser)
                    parsed_layers.append(name)
    return parser


def arguments_nn_layers(parser: argparse.ArgumentParser):

    # Retrieve layer specific arguments
    parser = layer_specific_args(parser)

    # activation and normalization arguments
    from cvnets.layers.activation import arguments_activation_fn
    parser = arguments_activation_fn(parser)

    from cvnets.layers.normalization import arguments_norm_layers
    parser = arguments_norm_layers(parser)

    return parser

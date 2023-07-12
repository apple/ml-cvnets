#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
import inspect
import os

from cvnets.layers.adaptive_pool import AdaptiveAvgPool2d
from cvnets.layers.base_layer import BaseLayer
from cvnets.layers.conv_layer import (
    ConvLayer1d,
    ConvLayer2d,
    ConvLayer3d,
    NormActLayer,
    SeparableConv1d,
    SeparableConv2d,
    SeparableConv3d,
    TransposeConvLayer2d,
)
from cvnets.layers.dropout import Dropout, Dropout2d
from cvnets.layers.embedding import Embedding
from cvnets.layers.flatten import Flatten
from cvnets.layers.global_pool import GlobalPool
from cvnets.layers.identity import Identity
from cvnets.layers.linear_attention import LinearSelfAttention
from cvnets.layers.linear_layer import GroupLinear, LinearLayer
from cvnets.layers.multi_head_attention import MultiHeadAttention
from cvnets.layers.normalization_layers import (
    AdjustBatchNormMomentum,
    get_normalization_layer,
    norm_layers_tuple,
)
from cvnets.layers.pixel_shuffle import PixelShuffle
from cvnets.layers.pooling import AvgPool2d, MaxPool2d
from cvnets.layers.positional_embedding import PositionalEmbedding
from cvnets.layers.single_head_attention import SingleHeadAttention
from cvnets.layers.softmax import Softmax
from cvnets.layers.stochastic_depth import StochasticDepth
from cvnets.layers.upsample import UpSample

__all__ = [
    "ConvLayer1d",
    "ConvLayer2d",
    "ConvLayer3d",
    "SeparableConv1d",
    "SeparableConv2d",
    "SeparableConv3d",
    "NormActLayer",
    "TransposeConvLayer2d",
    "LinearLayer",
    "GroupLinear",
    "GlobalPool",
    "Identity",
    "PixelShuffle",
    "UpSample",
    "MaxPool2d",
    "AvgPool2d",
    "Dropout",
    "Dropout2d",
    "AdjustBatchNormMomentum",
    "Flatten",
    "MultiHeadAttention",
    "SingleHeadAttention",
    "Softmax",
    "LinearSelfAttention",
    "Embedding",
    "PositionalEmbedding",
    "norm_layers_tuple",
    "StochasticDepth",
    "get_normalization_layer",
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

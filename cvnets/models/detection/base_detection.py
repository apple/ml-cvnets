#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import NamedTuple, Dict, Any, Tuple
import argparse

from utils import logger

from ..classification import BaseEncoder
from ... import parameter_list
from ...misc.init_utils import initialize_weights


DetectionPredTuple = NamedTuple(
    "DetectionPredTuple", [("labels", Any), ("scores", Any), ("boxes", Any)]
)


class BaseDetection(nn.Module):
    """
    Base class for the task of object detection
    """

    def __init__(self, opts, encoder: BaseEncoder) -> None:
        super().__init__()
        assert isinstance(encoder, BaseEncoder)
        self.encoder: BaseEncoder = encoder
        self.n_detection_classes = getattr(opts, "model.detection.n_classes", 80)

        enc_conf = self.encoder.model_conf_dict

        enc_ch_l5_out_proj = _check_out_channels(enc_conf, "exp_before_cls")
        enc_ch_l5_out = _check_out_channels(enc_conf, "layer5")
        enc_ch_l4_out = _check_out_channels(enc_conf, "layer4")
        enc_ch_l3_out = _check_out_channels(enc_conf, "layer3")
        enc_ch_l2_out = _check_out_channels(enc_conf, "layer2")
        enc_ch_l1_out = _check_out_channels(enc_conf, "layer1")

        self.enc_l5_channels = enc_ch_l5_out
        self.enc_l5_channels_exp = enc_ch_l5_out_proj
        self.enc_l4_channels = enc_ch_l4_out
        self.enc_l3_channels = enc_ch_l3_out
        self.enc_l2_channels = enc_ch_l2_out
        self.enc_l1_channels = enc_ch_l1_out

        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        return parser

    @staticmethod
    def reset_layer_parameters(layer, opts) -> None:
        """Initialize weights of a given layer"""
        initialize_weights(opts=opts, modules=layer.modules())

    def get_trainable_parameters(
        self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False
    ):
        """Returns a list of trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [1.0] * len(param_list)

    @staticmethod
    def profile_layer(layer, input: Tensor) -> Tuple[Tensor, float, float]:
        # profile a layer
        block_params = block_macs = 0.0
        if isinstance(layer, nn.Sequential):
            for layer_i in range(len(layer)):
                input, layer_param, layer_macs = layer[layer_i].profile_module(input)
                block_params += layer_param
                block_macs += layer_macs
                print(
                    "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                        layer[layer_i].__class__.__name__,
                        "Params",
                        round(layer_param / 1e6, 3),
                        "MACs",
                        round(layer_macs / 1e6, 3),
                    )
                )
        else:
            input, layer_param, layer_macs = layer.profile_module(input)
            block_params += layer_param
            block_macs += layer_macs
            print(
                "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                    layer.__class__.__name__,
                    "Params",
                    round(layer_param / 1e6, 3),
                    "MACs",
                    round(layer_macs / 1e6, 3),
                )
            )
        return input, block_params, block_macs

    def profile_model(self, input: Tensor):
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError


def _check_out_channels(config: Dict, layer_name: str) -> int:
    enc_ch_l: Dict = config.get(layer_name, None)
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

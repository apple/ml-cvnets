#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union
import argparse

from utils import logger

from ... import parameter_list
from ...layers import norm_layers_tuple
from ...misc.init_utils import initialize_weights


class BaseVideoEncoder(nn.Module):
    """
    Base class for the video backbones
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        self.round_nearest = 8
        self.model_conf_dict = dict()
        self.inference_mode = getattr(
            opts, "model.video_classification.inference_mode", False
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def reset_parameters(self, opts) -> None:
        """Reset parameters for all modules in the network"""
        initialize_weights(opts=opts, modules=self.modules())

    @staticmethod
    def reset_module_parameters(opts, module) -> None:
        """Reset parameters for a specific module in the network"""
        initialize_weights(opts=opts, modules=module)

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Dict:
        raise NotImplementedError

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict:
        raise NotImplementedError

    def extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers in the network"""
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """
        Get trainable parameters for the network
        """
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [1.0] * len(param_list)

    def profile_model(self, input: Tensor, *args, **kwargs) -> None:
        """
        This function computes FLOPs using fvcore (if installed).
        """
        logger.double_dash_line(dashes=65)
        print("{:>35} Summary".format(self.__class__.__name__))
        logger.double_dash_line(dashes=65)
        overall_params_py = sum([p.numel() for p in self.parameters()])

        try:
            from fvcore.nn import FlopCountAnalysis

            flop_analyzer = FlopCountAnalysis(self.eval(), input)
            flop_analyzer.unsupported_ops_warnings(False)
            total_flops = flop_analyzer.total()

            print(
                "Flops computed using FVCore for an input of size={} are {:>8.3f} G".format(
                    list(input.shape), total_flops / 1e9
                )
            )
        except ModuleNotFoundError:
            pass

        print(
            "{:<20} = {:>8.3f} M".format(
                "Overall parameters (sanity check)", overall_params_py / 1e6
            )
        )
        logger.double_dash_line(dashes=65)

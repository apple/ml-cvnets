#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
import argparse
from typing import Optional, Tuple, Dict

from cvnets import parameter_list


class BaseImageProjectionHead(nn.Module):
    """Base class that projects image representations to the same space as text representations"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.lr_mult = getattr(opts, "model.image_projection_head.lr_multiplier", 1.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model specific arguments"""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--model.image-projection-head.name",
            type=str,
            default=None,
            help="Name of the image projection head",
        )

        group.add_argument(
            "--model.image-projection-head.lr-multiplier",
            type=float,
            default=1.0,
            help="LR multiplier for image projection head",
        )

        return parser

    def reset_parameters(self) -> None:
        """Reset weights of a given layer"""
        raise NotImplementedError

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [self.lr_mult] * len(param_list)

    def forward(self, input: Dict, *args, **kwargs) -> Dict:
        raise NotImplementedError

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import argparse

from . import BaseImageProjectionHead, register_image_projection_head


@register_image_projection_head(name="simple_projection_nc2nc")
class SimpleImageProjectionHead(BaseImageProjectionHead):
    """This class implements simple projection head"""

    def __init__(self, opts, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        scale = in_dim**-0.5
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.feature_normalizaiton = not getattr(
            opts,
            "model.image_projection_head.simple_projection_nc2nc.no_feature_normalization",
            False,
        )

        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--model.image-projection-head.simple-projection-nc2nc.no-feature-normalization",
            action="store_true",
            help="Don't normalize image features",
        )

        return parser

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 2
        ), "Input should be 2-dimensional (Batch x in_dim). Got: {}".format(x.shape)

        # [batch, in_dim] x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        if self.feature_normalizaiton:
            x = F.normalize(x, dim=-1)
        return x

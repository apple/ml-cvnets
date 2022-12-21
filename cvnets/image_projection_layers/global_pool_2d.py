#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from cvnets.layers import GlobalPool
from . import BaseImageProjectionHead, register_image_projection_head


@register_image_projection_head(name="global_pool_nchw2nc")
class GlobalPool2D(BaseImageProjectionHead):
    """This class implements global pooling with linear projection"""

    def __init__(self, opts, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        scale = in_dim**-0.5
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.feature_normalization = not getattr(
            opts,
            "model.image_projection_head.global_pool_nchw2nc.no_feature_normalization",
            False,
        )

        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--model.image-projection-head.global-pool-nchw2nc.no-feature-normalization",
            action="store_true",
            help="Don't normalize image features",
        )

        return parser

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 4
        ), "Input should be 4-dimensional (Batch x in_dim x in_height x in_width). Got: {}".format(
            x.shape
        )

        # [batch, in_dim, in_height, in_width] --> [batch, in_dim]
        x = self.pool(x)
        # [batch, in_dim]  x [in_dim, out_dim] --> [batch, out_dim]
        x = x @ self.proj
        if self.feature_normalization:
            x = F.normalize(x, dim=-1)
        return x

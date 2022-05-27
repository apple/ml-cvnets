#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Tuple, Optional

from .base_layer import BaseLayer
from .linear_layer import LinearLayer
from .dropout import Dropout
from ..misc.profiler import module_profile


class SingleHeadAttention(BaseLayer):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5

    def __repr__(self) -> str:
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def forward(self, x: Tensor) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        qkv = self.qkv_proj(x)

        # [N, P, 3C] --> [N, P, C] x 3
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        # number of operations in QK^T
        m_qk = (seq_len * in_channels * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * in_channels * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.out_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        return input, params, macs

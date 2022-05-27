#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from torch.nn import functional as F

from utils import logger

from .base_layer import BaseLayer
from .linear_layer import LinearLayer
from .dropout import Dropout
from ..misc.profiler import module_profile


class MultiHeadAttention(BaseLayer):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
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
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_tracing(self, x_q: Tensor, x_kv: Optional[Tensor] = None) -> Tensor:
        if x_kv is None:
            # [N, P, C] --> # [N, P, 3C]
            qkv = self.qkv_proj(x_q)
            # # [N, P, 3C] --> # [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            # [N, P, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, P, c] x h, where C = c * h
        query = torch.chunk(query, chunks=self.num_heads, dim=-1)
        value = torch.chunk(value, chunks=self.num_heads, dim=-1)
        key = torch.chunk(key, chunks=self.num_heads, dim=-1)

        wt_out = []
        for h in range(self.num_heads):
            attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))
            attn_h = self.softmax(attn_h)
            attn_h = self.attn_dropout(attn_h)
            out_h = torch.matmul(attn_h, value[h])
            wt_out.append(out_h)

        wt_out = torch.cat(wt_out, dim=-1)
        wt_out = self.out_proj(wt_out)
        return wt_out

    def forward_default(self, x_q: Tensor, x_kv: Optional[Tensor] = None) -> Tensor:

        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, P, C] --> [N, P, 3C] --> [N, P, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)
            # [N, P, 3, h, c] --> [N, h, 3, P, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, P, C] --> [N, h, P, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            # cross-attention
            # [N, P, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )
            # [N, P, C] --> [N, P, h, c] --> [N, h, P, c]
            query = (
                query.reshape(b_sz, n_patches, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            # [N, P, 2C] --> [N, P, 2, h, c]
            kv = kv.reshape(b_sz, n_patches, 2, self.num_heads, self.head_dim)
            # [N, P, 2, h, c] --> [N, h, 2, P, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, P, c] --> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] --> [N, h, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] --> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] --> [N, P, h, c] --> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

    def forward(
        self, x_q: Tensor, x_kv: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if self.coreml_compatible:
            return self.forward_tracing(x_q=x_q, x_kv=x_kv)
        else:
            return self.forward_default(x_q=x_q, x_kv=x_kv)

    def profile_module(self, input) -> Tuple[Tensor, float, float]:
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        # number of operations in QK^T
        m_qk = (seq_len * seq_len * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * seq_len * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.out_proj, x=input)
        params += p
        macs += m * seq_len * b_sz

        return input, params, macs

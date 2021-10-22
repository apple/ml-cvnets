#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Tuple, Optional
from sys import platform

from .base_layer import BaseLayer
from .linear_layer import LinearLayer
from .dropout import Dropout
from ..misc.profiler import module_profile


class MultiHeadAttention(BaseLayer):
    '''
            This layer applies a multi-head attention as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    '''
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: Optional[float] =0.0, 
                 bias: Optional[bool] = True,
                 *args, **kwargs):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param attn_dropout: Attention dropout
        :param bias: Bias
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Got: embed_dim={} and num_heads={}".format(embed_dim, num_heads)

        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3*embed_dim, bias=bias)

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.mac_device = False
        if platform == "darwin":
            self.mac_device = True

    def forward_mac_device(self, x: Tensor) -> Tensor:
        # [B x N x C]
        qkv = self.qkv_proj(x)

        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

        query = query * self.scaling

        # [B x N x C] --> [B x N x c] x h
        query = torch.chunk(query, chunks=self.num_heads, dim=-1)
        value = torch.chunk(value, chunks=self.num_heads, dim=-1)
        key = torch.chunk(key, chunks=self.num_heads, dim=-1)

        wt_out = []
        for h in range(self.num_heads):
            attn_h = torch.bmm(query[h], key[h].transpose(1, 2))
            attn_h = self.softmax(attn_h)
            attn_h = self.attn_dropout(attn_h)
            out_h = torch.bmm(attn_h, value[h])
            wt_out.append(out_h)

        wt_out = torch.cat(wt_out, dim=-1)
        wt_out = self.out_proj(wt_out)
        return wt_out

    def forward_other(self, x: Tensor) -> Tensor:
        # [B x N x C]
        b_sz, n_patches, in_channels = x.shape

        # [B x N x C] --> [B x N x 3 x h x C]
        qkv = (
            self.qkv_proj(x)
                .reshape(b_sz, n_patches, 3, self.num_heads, -1)
        )
        # [B x N x 3 x h x C] --> [B x h x 3 x N x C]
        qkv = qkv.transpose(1, 3)

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [B x h x N x C] --> [B x h x c x N]
        key = key.transpose(2, 3)

        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [B x h x N x N] x [B x h x N x c] --> [B x h x N x c]
        out = torch.matmul(attn, value)

        # [B x h x N x c] --> [B x N x h x c] --> [B x N x C=ch]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        if self.mac_device:
            return self.forward_mac_device(x)
        else:
            return self.forward_other(x)

    def profile_module(self, input) -> (Tensor, float, float):
        b_sz, seq_len, in_channels = input.shape
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += (m * seq_len * b_sz)

        # number of operations in QK^T
        m_qk = (seq_len * in_channels * in_channels) * b_sz
        macs += m_qk

        # number of operations in computing weighted sum
        m_wt = (seq_len * in_channels * in_channels) * b_sz
        macs += m_wt

        out_p, p, m = module_profile(module=self.out_proj, x=input)
        params += p
        macs += (m * seq_len * b_sz)

        return input, params, macs

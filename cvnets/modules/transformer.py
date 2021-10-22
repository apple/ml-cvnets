#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional

from ..layers import get_normalization_layer, LinearLayer, get_activation_fn, MultiHeadAttention, Dropout
from ..modules import BaseModule
from ..misc.profiler import module_profile


class TransformerEncoder(BaseModule):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """
    def __init__(self, opts, embed_dim: int, ffn_latent_dim: int, num_heads: Optional[int] = 8, attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1, ffn_dropout: Optional[float] = 0.0,
                 transformer_norm_layer: Optional[str] = "layer_norm",
                 *args, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(opts=opts),
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x: Tensor) -> Tensor:

        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        b_sz, seq_len = input.shape[:2]

        out, p_mha, m_mha = module_profile(module=self.pre_norm_mha, x=input)

        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)
        m_ffn = (m_ffn * b_sz * seq_len)

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs

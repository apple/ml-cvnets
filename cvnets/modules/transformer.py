#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Union, Tuple

from ..layers import (
    get_normalization_layer,
    LinearLayer,
    get_activation_fn,
    ConvLayer,
    MultiHeadAttention,
    Dropout,
    SingleHeadAttention,
    LinearSelfAttention,
)

from ..modules import BaseModule
from ..misc.profiler import module_profile


class TransformerEncoder(BaseModule):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (Optional[int]) : Number of heads in multi-head attention. Default: 8
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
                coreml_compatible=getattr(
                    opts, "common.enable_coreml_compatible_module", False
                ),
            )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        act_name = self.build_act_layer(opts=opts)
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = self.pre_norm_mha[0](x)  # norm
        x = self.pre_norm_mha[1](x_q=x, x_kv=x_prev)  # mha
        x = self.pre_norm_mha[2](x)  # dropout
        x = x + res

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        b_sz, seq_len = input.shape[:2]

        out, p_mha, m_mha = module_profile(module=self.pre_norm_mha, x=input)

        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)
        m_ffn = m_ffn * b_sz * seq_len

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs


# TODO: Add link to MobileViTv2 paper


class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            ConvLayer(
                opts=opts,
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            Dropout(p=ffn_dropout),
            ConvLayer(
                opts=opts,
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, norm_layer={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.norm_name,
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out, p_mha, m_mha = module_profile(module=self.pre_norm_attn, x=input)
        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs

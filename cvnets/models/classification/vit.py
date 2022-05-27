#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import argparse

from utils import logger

from . import register_cls_models, BaseEncoder
from .config.vit import get_configuration
from ...layers import (
    ConvLayer,
    LinearLayer,
    get_normalization_layer,
    SinusoidalPositionalEncoding,
    Dropout,
    LearnablePositionEncoding,
)
from ...modules import TransformerEncoder


@register_cls_models(name="vit")
class VisionTransformer(BaseEncoder):
    """
    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Use sinusoidal positional encoding, allowing us to use ViT with any input size
        3. Do not use DropoutPath
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        image_channels = 3
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        super().__init__(*args, **kwargs)

        vit_config = get_configuration(opts)

        patch_size = vit_config["patch_size"]
        embed_dim = vit_config["embed_dim"]
        ffn_dim = vit_config["ffn_dim"]
        pos_emb_drop_p = vit_config["pos_emb_drop_p"]
        n_transformer_layers = vit_config["n_transformer_layers"]
        num_heads = vit_config["n_attn_heads"]
        attn_dropout = vit_config["attn_dropout"]
        dropout = vit_config["dropout"]
        ffn_dropout = vit_config["ffn_dropout"]
        norm_layer = vit_config["norm_layer"]

        kernel_size = patch_size
        if patch_size % 2 == 0:
            kernel_size += 1

        self.patch_emb = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=True,
            use_norm=False,
            use_act=False,
        )

        use_cls_token = not getattr(
            opts, "model.classification.vit.no_cls_token", False
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        transformer_blocks = [
            TransformerEncoder(
                opts=opts,
                embed_dim=embed_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=norm_layer,
            )
            for _ in range(n_transformer_layers)
        ]
        transformer_blocks.append(
            get_normalization_layer(
                opts=opts, num_features=embed_dim, norm_type=norm_layer
            )
        )

        self.transformer = nn.Sequential(*transformer_blocks)
        self.classifier = LinearLayer(embed_dim, num_classes)

        self.reset_parameters(opts=opts)

        if self.cls_token is not None:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        vocab_size = getattr(opts, "model.classification.vit.vocab_size", 1000)
        if getattr(opts, "model.classification.vit.learnable_pos_emb", False):
            self.pos_embed = LearnablePositionEncoding(
                num_embeddings=vocab_size,
                embed_dim=embed_dim,
                dropout=pos_emb_drop_p,
                channels_last=True,
            )
            nn.init.normal_(
                self.pos_embed.pos_emb.weight, mean=0, std=embed_dim ** -0.5
            )
        else:
            self.pos_embed = SinusoidalPositionalEncoding(
                d_model=embed_dim,
                dropout=pos_emb_drop_p,
                channels_last=True,
                max_len=vocab_size,
            )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.vit.mode",
            type=str,
            default="tiny",
            help="ViT mode. Default is Tiny",
        )
        group.add_argument(
            "--model.classification.vit.dropout",
            type=float,
            default=0.0,
            help="Dropout in ViT layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.vit.vocab-size",
            type=int,
            default=1000,
            help="Vocab size (or max patches) in ViT. Defaults to 1000",
        )
        group.add_argument(
            "--model.classification.vit.learnable-pos-emb",
            action="store_true",
            help="Use learnable positional encoding instead of sinusiodal",
        )
        group.add_argument(
            "--model.classification.vit.no-cls-token",
            action="store_true",
            help="Do not use classificaiton token",
        )
        return parser

    def extract_patch_embeddings(self, x: Tensor) -> Tensor:
        # x -> [B, C, H, W]
        B_ = x.shape[0]

        # [B, C, H, W] --> [B, C, n_h, n_w]
        patch_emb = self.patch_emb(x)
        # [B, C, n_h, n_w] --> [B, C, N]
        patch_emb = patch_emb.flatten(2)
        # [B, C, N] --> [B, N, C]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        # add classification token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B_, -1, -1)
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        patch_emb = self.pos_embed(patch_emb)
        return patch_emb

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.extract_patch_embeddings(x)

        x = self.transformer(x)

        # grab the first token and classify
        if self.cls_token is not None:
            x = self.classifier(x[:, 0])
        else:
            x = torch.mean(x, dim=1)
            x = self.classifier(x)
        return x

    def profile_model(self, input: Tensor, *args, **kwargs) -> None:
        logger.log("Model statistics for an input of size {}".format(input.size()))
        logger.double_dash_line(dashes=65)
        print("{:>35} Summary".format(self.__class__.__name__))
        logger.double_dash_line(dashes=65)

        out_dict = {}
        overall_params, overall_macs = 0.0, 0.0
        patch_emb, overall_params, overall_macs = self._profile_layers(
            self.patch_emb,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        patch_emb = patch_emb.flatten(2)

        # [B, C, N] --> [B, N, C]
        patch_emb = patch_emb.transpose(1, 2)

        if self.cls_token is not None:
            # add classification token
            cls_tokens = self.cls_token.expand(patch_emb.shape[0], -1, -1)
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        patch_emb, overall_params, overall_macs = self._profile_layers(
            self.transformer,
            input=patch_emb,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )

        patch_emb, overall_params, overall_macs = self._profile_layers(
            self.classifier,
            input=patch_emb[:, 0],
            overall_params=overall_params,
            overall_macs=overall_macs,
        )

        logger.double_dash_line(dashes=65)
        print("{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6))
        # Counting Addition and Multiplication as 1 operation
        print("{:<20} = {:>8.3f} M".format("Overall MACs", overall_macs / 1e6))
        overall_params_py = sum([p.numel() for p in self.parameters()])
        print(
            "{:<20} = {:>8.3f} M".format(
                "Overall parameters (sanity check)", overall_params_py / 1e6
            )
        )
        logger.double_dash_line(dashes=65)

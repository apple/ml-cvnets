# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn

from cvnets.layers import (
    Dropout,
    Embedding,
    PositionalEmbedding,
    get_normalization_layer,
)
from cvnets.modules import TransformerEncoder
from cvnets.text_encoders import TEXT_ENCODER_REGISTRY, BaseTextEncoder
from utils import logger


@TEXT_ENCODER_REGISTRY.register(name="transformer")
class TextTransformer(BaseTextEncoder):
    def __init__(self, opts, projection_dim: int, *args, **kwargs) -> None:
        model_dim = getattr(opts, "model.text.transformer.model_dim", 512)
        no_scale_embedding = getattr(
            opts, "model.text.transformer.no_scale_embedding", False
        )
        no_pos_embedding = getattr(
            opts, "model.text.transformer.no_pos_embedding", False
        )
        embed_dropout = getattr(opts, "model.text.transformer.embed_dropout", 0.0)
        dropout = getattr(opts, "model.text.transformer.dropout", 0.0)
        attn_dropout = getattr(opts, "model.text.transformer.attn_dropout", 0.0)
        ffn_dropout = getattr(opts, "model.text.transformer.ffn_dropout", 0.0)
        norm_layer = getattr(opts, "model.text.transformer.norm_layer", None)

        gradient_ckpt = getattr(
            opts, "model.text.transformer.gradient_checkpoint", False
        )

        if norm_layer is None:
            logger.error(
                "Normalization layer can not be None in {}".format(
                    self.__class__.__name__
                )
            )

        super().__init__(opts=opts, projection_dim=projection_dim, *args, **kwargs)

        # token embedding layer
        padding_index = getattr(opts, "dataset.padding_index", None)
        self.embedding_layer = Embedding(
            opts=opts,
            embedding_dim=model_dim,
            padding_idx=padding_index,
            num_embeddings=self.vocab_size,
        )
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        context_length = getattr(opts, "dataset.text_context_length")

        if getattr(opts, "common.debug_mode", False):
            context_length = 77

        assert context_length is not None, (
            "Context length can't be None. Please set dataset.text_context_length "
            "argument in your dataset class"
        )

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(
                opts=opts,
                num_embeddings=context_length,
                embedding_dim=model_dim,
                padding_idx=getattr(opts, "dataset.padding_index", None),
                is_learnable=not getattr(
                    opts, "model.text.transformer.sinusoidal_pos_emb", False
                ),
            )
        )

        self.embedding_dropout = Dropout(p=embed_dropout)

        # Transformer layer

        n_transformer_layers = getattr(
            opts, "model.text.transformer.n_transformer_layers", 6
        )
        # FFN multipliers for transformer layer
        ffn_multipliers = getattr(
            opts, "model.text.transformer.ffn_multiplier_per_layer", 4.0
        )
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        if not isinstance(ffn_multipliers, Sequence):
            logger.error(
                "{} expects FFN multipliers as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(ffn_multipliers)
                )
            )
        elif (
            isinstance(ffn_multipliers, Sequence)
            and len(ffn_multipliers) != n_transformer_layers
        ):
            logger.error(
                "We need FFN multiplier for each transformer layer. Got {} ffn multipliers while number of "
                "transformer layers = {}".format(
                    len(ffn_multipliers), n_transformer_layers
                )
            )
        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0)
            for ffn_mult in ffn_multipliers
        ]

        # Heads for transformer layers
        mha_heads = getattr(opts, "model.text.transformer.n_heads_per_layer", 8)
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if not isinstance(mha_heads, Sequence):
            logger.error(
                "{} expects MHA heads as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(mha_heads)
                )
            )
        elif isinstance(mha_heads, Sequence) and len(mha_heads) != n_transformer_layers:
            logger.error(
                "{} needs MHA heads for each transformer layer. Got {} mha heads while number of "
                "transformer layers = {}".format(
                    self.__class__.__name__, len(mha_heads), n_transformer_layers
                )
            )

        self.transformer = nn.ModuleList(
            [
                TransformerEncoder(
                    opts=opts,
                    embed_dim=model_dim,
                    num_heads=mha_heads[layer_idx],
                    ffn_latent_dim=ffn_dims[layer_idx],
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    dropout=dropout,
                    transformer_norm_layer=norm_layer,
                )
                for layer_idx in range(n_transformer_layers)
            ]
        )
        self.final_layer_norm = get_normalization_layer(
            opts, num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = nn.Parameter(
            torch.empty(model_dim, self.projection_dim)
        )
        self.model_dim = model_dim
        self.reset_parameters_clip_style()
        self.gradient_ckpt = gradient_ckpt
        self.use_pytorch_mha = False
        self.causal_masking = getattr(
            opts, "model.text.transformer.causal_masking", False
        )
        self.classes_per_split_zero_shot = max(
            1,
            int(getattr(opts, "model.text.transformer.classes_per_split_zero_shot", 1)),
        )

    def reset_parameters_clip_style(self):
        """This function resets the weights of Transformer model as done in the CLIP paper"""

        # reset the weights of the embedding and positional embedding layers
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
        # if self.positional_embedding is not None and not getattr(
        #    self.opts, "model.text.transformer.sinusoidal_pos_emb", False
        # ):
        #    nn.init.normal_(
        #        self.positional_embedding.pos_embed.weight, mean=0.0, std=0.01
        #    )

        # compute standard deviation for different linear layers in transformer model
        attn_std = self.model_dim**-0.5
        proj_std = attn_std * ((2 * len(self.transformer)) ** -0.5)
        fc_std = (2 * self.model_dim) ** -0.5

        for block in self.transformer:
            # multi-head attention QKV projection layer
            nn.init.normal_(
                block.pre_norm_mha[1].qkv_proj.weight, mean=0.0, std=attn_std
            )
            # multi-head attention output projection layer
            nn.init.normal_(
                block.pre_norm_mha[1].out_proj.weight, mean=0.0, std=proj_std
            )
            # FFN expansion layer
            nn.init.normal_(block.pre_norm_ffn[1].weight, mean=0.0, std=fc_std)
            # FFN reduction layer
            nn.init.normal_(block.pre_norm_ffn[4].weight, mean=0.0, std=proj_std)

        nn.init.normal_(self.projection_layer, mean=0.0, std=attn_std)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != TextTransformer:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.text.transformer.model-dim",
            type=int,
            default=512,
            help="Model dimension of the transformer model",
        )

        group.add_argument(
            "--model.text.transformer.no-scale-embedding",
            action="store_true",
            help="Do not scale the output of embedding layer in {}".format(
                cls.__name__
            ),
        )

        group.add_argument(
            "--model.text.transformer.no-pos-embedding",
            action="store_true",
            help="Do not add positional embeddings to the output of embedding layer in {}".format(
                cls.__name__
            ),
        )

        group.add_argument(
            "--model.text.transformer.embed-dropout",
            type=float,
            default=0.0,
            help="Dropout in embedding layer",
        )

        # transformer layer parameters
        default_layers = 6
        group.add_argument(
            "--model.text.transformer.n-transformer-layers",
            type=int,
            default=default_layers,
            help="Number of transformer layers in {}".format(cls.__name__),
        )
        group.add_argument(
            "--model.text.transformer.n-heads-per-layer",
            type=int,
            default=[8] * default_layers,
            nargs="+",
            help="Number of transformer heads per transformer layer",
        )

        group.add_argument(
            "--model.text.transformer.ffn-multiplier-per-layer",
            type=float,
            default=[4.0] * default_layers,
            nargs="+",
            help="FFN multiplier for each transformer layer",
        )
        group.add_argument(
            "--model.text.transformer.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in multi-head attention",
        )
        group.add_argument(
            "--model.text.transformer.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between linear layers in FFN",
        )
        group.add_argument(
            "--model.text.transformer.dropout",
            type=float,
            default=0.0,
            help="Dropout in transformer",
        )

        group.add_argument(
            "--model.text.transformer.norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer",
        )

        group.add_argument(
            "--model.text.transformer.sinusoidal-pos-emb",
            action="store_true",
            help="Use sinusoidal positional embedding",
        )

        group.add_argument(
            "--model.text.transformer.gradient-checkpoint",
            action="store_true",
            help="Use gradient checkpointing",
        )
        group.add_argument(
            "--model.text.transformer.num-checkpoint-segments",
            type=int,
            default=1,
            help="Number of gradient checkpoint segments",
        )

        group.add_argument(
            "--model.text.transformer.causal-masking",
            action="store_true",
            help="Use causal masking",
        )

        group.add_argument(
            "--model.text.transformer.classes-per-split-zero-shot",
            type=int,
            default=20,
            help="Divide zero-shot classes into these many chunks, for faster processing",
        )

        return parser

    def forward_embedding(
        self,
        text_tokens: Tensor,
    ):
        # [Batch, Seq_len] --> [Batch, Seq_len, hidden_dim]
        token_emb = self.embedding_layer(text_tokens)
        # token_emb = self.embed_scale * token_emb
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(
                token_emb.dtype
            )
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        if not self.use_pytorch_mha:
            mask = mask.unsqueeze(0)  # add dummy batch dimension
            mask = mask.expand(batch_size, -1, -1)
        return mask

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        """
        Returns token embeddings.

        :param text_tokens: a tensor of token indices. ([Batch, Seq_len])
        :param key_padding_mask: a tensor of boolean values as the padding mask.
        :param return_all_tokens: a boolean flag to return all tokens, defaults to False
            to return only EOT token embedding.
        :return: a tensor of [Batch, Seq_len, hidden_dim] if return_all_tokens is True,
            otherwise a tensor of [Batch, hidden_dim].
        """
        # discrete tokens to continuous embeddings
        # [Batch, Seq_len] --> [Batch, Seq_len, hidden_dim]
        token_emb = self.forward_embedding(text_tokens)

        # [1, Seq_len, Seq_len]
        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
            key_padding_mask = None

        if self.use_pytorch_mha:
            # [Batch, Seq_len, hidden_dim] --> [Seq_len, Batch, hidden_dim]
            # we will use PyTorch's multi-head attention, which uses sequence_first format
            token_emb = token_emb.transpose(0, 1)

        for layer in self.transformer:
            if self.gradient_ckpt:
                token_emb = gradient_checkpoint_fn(
                    layer, token_emb, None, key_padding_mask, attn_mask
                )
            else:
                token_emb = layer(
                    token_emb,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    use_pytorch_mha=self.use_pytorch_mha,
                )

        # Apply layer norm
        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            if self.use_pytorch_mha:
                # [Seq_len, Batch, hidden_dim] --> [Batch, Seq_len, hidden_dim]
                token_emb = token_emb.transpose(0, 1)

            return token_emb

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.use_pytorch_mha:
            token_emb = token_emb[
                text_tokens.argmax(dim=-1), torch.arange(text_tokens.shape[0])
            ]
        else:
            token_emb = token_emb[
                torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
            ]

        token_emb = token_emb @ self.projection_layer
        # normalize text features
        token_emb = F.normalize(token_emb, dim=-1)
        return token_emb

    def forward_zero_shot(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        # In case of zero-shot evaluation, text tokens is of shape [Batch, num_classes, num_captions, context_length]
        # For example, in the ImageNet dataset, we have 1000 classes, and for each class we generate certain number of
        # captions (each caption with context_length tokens)

        if self.training:
            raise NotImplementedError(
                "Zero-shot evaluation is only supported with eval mode"
            )

        if text_tokens.ndim != 4:
            logger.error(
                "For zero-shot evaluation, expected size of text is [Batch, Num_classes, num_captions, context_len]"
            )

        batch_size, num_classes, num_captions, context_len = text_tokens.shape

        # for zero-shot evaluation, text templates are the same across all images in the batch
        # Therefore, batch size should be 1.
        if batch_size > 1:
            text_tokens = text_tokens[0:1]
            batch_size = 1
            logger.warning(
                "For zero-shot evaluation, text templates are the same across all images in the batch."
                "Got: {}. Please consider adjusting collate function.".format(
                    batch_size
                )
            )

        text_features = []

        for start_idx in range(0, num_classes, self.classes_per_split_zero_shot):
            end_idx = min(start_idx + self.classes_per_split_zero_shot, num_classes)

            text_tokens_split = text_tokens[0, start_idx:end_idx, ...]
            num_classes_split = text_tokens_split.shape[0]
            text_tokens_split = text_tokens_split.reshape(
                num_classes_split * num_captions, context_len
            )

            key_padding_mask_split = None
            if key_padding_mask is not None:
                key_padding_mask_split = key_padding_mask[0, start_idx:end_idx, ...]
                key_padding_mask_split = key_padding_mask_split.reshape(
                    num_classes_split * num_captions, context_len
                )

            # [num_classes_per_split * num_cations, context_len] --> [num_classes_per_split * num_cations, latent_dim]
            class_embedding_split = self.encode_text(
                text_tokens=text_tokens_split, key_padding_mask=key_padding_mask_split
            )

            # [num_classes_per_split * num_cations, latent_dim] --> [num_classes_per_split, num_cations, latent_dim]
            class_embedding_split = class_embedding_split.reshape(
                num_classes_split, num_captions, class_embedding_split.shape[-1]
            )

            # Compute mean of all classes
            # [num_classes_per_split, num_cations, latent_dim] --> [num_classes_per_split, latent_dim]
            mean_class_embedding_split = class_embedding_split.mean(dim=1)

            # Normalize the embeddings
            mean_class_embedding_split = F.normalize(mean_class_embedding_split, dim=-1)

            text_features.append(mean_class_embedding_split)

        # [num_classes_per_split, latent_dim] * num_splits --> [num_classes, Latent_dim]
        text_features = torch.cat(text_features, dim=0)
        # [num_classes, Latent_dim] --> [Latent_dim, num_classes]
        text_features = text_features.transpose(0, 1)
        return text_features.contiguous()

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:

        if text_tokens.dim() == 4:
            # It's for zero-shot evaluation.
            # Each class in the dataset has multiple captions
            # Encoding happens separately for each classes/captions due to OOM issue
            return self.forward_zero_shot(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs
            )
        elif text_tokens.dim() == 2:
            # Image-text pair data with single caption
            # [B, CL] --> [B, d]
            text_tokens = self.encode_text(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs
            )
            return text_tokens
        elif text_tokens.dim() == 3:
            # Image-text pair with multiple captions per image (e.g. Flickr-30k)
            # Treat them as separate captions by reshaping into batch dim
            # [B, N, C] --> [B*N, C] -encode-> [B*N, d] --> [B, N, d]
            b, n, _ = text_tokens.shape
            text_tokens = text_tokens.reshape(b * n, -1)
            if key_padding_mask:
                key_padding_mask = key_padding_mask.reshape(b * n, -1)
            text_tokens = self.encode_text(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs
            )
            text_tokens = text_tokens.reshape(b, n, -1)
            return text_tokens
        else:
            raise NotImplementedError

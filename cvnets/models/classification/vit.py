#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential as gradient_checkpoint_fn

from cvnets.layers import (
    ConvLayer2d,
    Dropout,
    Identity,
    LinearLayer,
    MaxPool2d,
    PositionalEmbedding,
    TransposeConvLayer2d,
    get_normalization_layer,
)
from cvnets.misc.common import parameter_list
from cvnets.misc.init_utils import initialize_conv_layer
from cvnets.models import MODEL_REGISTRY
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from cvnets.models.classification.config.vit import get_configuration
from cvnets.modules import TransformerEncoder
from utils import logger


@MODEL_REGISTRY.register(name="vit", type="classification")
class VisionTransformer(BaseImageEncoder):
    """
    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_. Our model implementation
    is inspired from `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Our positional encoding implementation allows us to use ViT with any multiple input scales
        3. We do not use StochasticDepth
        4. We do not add positional encoding to class token (if enabled), as suggested in `DeiT-3 paper <https://arxiv.org/abs/2204.07118>`_
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        image_channels = 3
        num_classes = getattr(opts, "model.classification.n_classes")
        if num_classes is None:
            logger.error(
                "Please specify number of classes using --model.classification.n-classes argument"
            )
        pytorch_mha = getattr(opts, "model.classification.vit.use_pytorch_mha")

        super().__init__(opts, *args, **kwargs)
        if pytorch_mha and self.gradient_checkpointing:
            logger.error(
                "Current version of ViT supports PyTorch MHA without gradient checkpointing. "
                "Please use either of them, but not both"
            )

        # If output stride is not None, then it is likely a segmentation task.
        # in that case, ensure that output stride is either 8 or 16
        kernel_sizes_conv_stem = [4, 2, 2]
        strides_conv_stem = [4, 2, 2]
        if self.output_stride is not None and self.output_stride not in [8, 16]:
            logger.error("Output stride should be 8 or 16")
        elif self.output_stride is not None and self.output_stride == 8:
            strides_conv_stem[0] = 2

        vit_config = get_configuration(opts)

        # Typically, in the ImageNet dataset, we use 224x224 as a resolution.
        # For out ViT implementation, patch size is 16 (16 = 4 * 2 * 2)
        # Therefore, total number of embeddings along width and height are (224 / 16)^2
        num_embeddings = (224 // 16) ** 2

        embed_dim = vit_config["embed_dim"]
        ffn_dim = vit_config["ffn_dim"]
        pos_emb_drop_p = vit_config["pos_emb_drop_p"]
        n_transformer_layers = vit_config["n_transformer_layers"]
        num_heads = vit_config["n_attn_heads"]
        attn_dropout = vit_config["attn_dropout"]
        dropout = vit_config["dropout"]
        ffn_dropout = vit_config["ffn_dropout"]
        norm_layer = vit_config["norm_layer"]

        conv_stem_proj_dim = max(32, embed_dim // 4)
        patch_emb = [
            ConvLayer2d(
                opts=opts,
                in_channels=image_channels,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[0],
                stride=strides_conv_stem[0],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[1],
                stride=strides_conv_stem[1],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=embed_dim,
                kernel_size=kernel_sizes_conv_stem[2],
                stride=strides_conv_stem[2],
                bias=True,
                use_norm=False,
                use_act=False,
            ),
        ]

        self.patch_emb = nn.Sequential(*patch_emb)

        use_cls_token = not getattr(opts, "model.classification.vit.no_cls_token")
        stochastic_dropout = getattr(
            opts, "model.classification.vit.stochastic_dropout"
        )
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]
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
                stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
            )
            for layer_idx in range(n_transformer_layers)
        ]

        self.post_transformer_norm = get_normalization_layer(
            opts=opts, num_features=embed_dim, norm_type=norm_layer
        )

        self.transformer = nn.Sequential(*transformer_blocks)
        self.classifier = LinearLayer(embed_dim, num_classes)

        self.reset_parameters(opts=opts)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = PositionalEmbedding(
            opts=opts,
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            is_learnable=not getattr(
                opts, "model.classification.vit.sinusoidal_pos_emb"
            ),
            interpolation_mode="bilinear",
        )
        self.emb_dropout = Dropout(p=pos_emb_drop_p)
        self.use_pytorch_mha = pytorch_mha
        self.embed_dim = embed_dim
        # We need to enable gradient checkpointing (--model.classification.gradient-checkpointing)
        # to use checkpoint segments in ViT
        self.checkpoint_segments = getattr(
            opts, "model.classification.vit.checkpoint_segments"
        )

        self.model_conf_dict = {
            "conv1": {"in": image_channels, "out": embed_dim},
            "layer1": {"in": embed_dim, "out": embed_dim},
            "layer2": {"in": embed_dim, "out": embed_dim},
            "layer3": {"in": embed_dim, "out": embed_dim},
            "layer4": {"in": embed_dim, "out": embed_dim},
            "layer5": {"in": embed_dim, "out": embed_dim},
            "exp_before_cls": {"in": embed_dim, "out": embed_dim},
            "cls": {"in": embed_dim, "out": num_classes},
        }

        use_simple_fpn = getattr(opts, "model.classification.vit.use_simple_fpn")
        self.simple_fpn = None
        if use_simple_fpn:
            # for object detection, we add Simple FPN on top of ViT backbone, so that it can
            # generate multi-scale representations. See https://arxiv.org/abs/2203.16527 for details
            self.simple_fpn = self._build_simple_fpn_layers(opts, embed_dim, norm_layer)
            self.reset_simple_fpn_params()

        self.update_layer_norm_eps()

    def update_layer_norm_eps(self):
        # Most ViT models use LayerNorm with 10^-6 eps. So, we update it here
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = 1e-6

    def reset_simple_fpn_params(self) -> None:
        # reset simple FPN parameters
        if self.simple_fpn is not None:
            for m in self.simple_fpn.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    initialize_conv_layer(m, init_method="kaiming_uniform")

    def _apply_layer_wise_lr(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[List, List]:
        """
        This function adjusts the learning rate of each layer in transformer module.
        Layer-wise learning is a bit involved and requires a knowledge of how each layer is consumed
        during the forward pass. We adjust the learning rate of patch embedding and transformer layers
        while keeping the classifier and SimpleFPN at 1.0. This is because layer_wise_lr is typically
        applied during fine-tuning for down-stream tasks.

        For ViT (classification tasks), the path is like this:
        Patch Embedding --> Transformer --> PostNorm --> Classifier

        For ViT (detection tasks), the path is like this:
        Patch Embedding --> Transformer --> PostNorm --> SimpleFPN

        """
        n_layers = 1 + len(self.transformer)
        layer_wise_lr = [
            round(self.layer_wise_lr_decay_rate ** (n_layers - i), 5)
            for i in range(n_layers)
        ]
        module_name = kwargs.pop("module_name", "")

        param_list = []
        param_lr_list = []

        if self.neural_augmentor:
            neural_aug_params = parameter_list(
                named_parameters=self.neural_augmentor.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "neural_augmentor.",
                *args,
                **kwargs,
            )
            param_list.extend(neural_aug_params)
            param_lr_list.extend([layer_wise_lr[0]] * len(neural_aug_params))

        # Patch embedding related parameters
        embedding_params = parameter_list(
            named_parameters=self.patch_emb.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "patch_emb.",
            *args,
            **kwargs,
        )
        param_list.extend(embedding_params)
        param_lr_list.extend([layer_wise_lr[0]] * len(embedding_params))

        # positional embedding parameters
        pos_emb_params = parameter_list(
            named_parameters=self.pos_embed.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "pos_embed.",
            *args,
            **kwargs,
        )
        param_list.extend(pos_emb_params)
        param_lr_list.extend([layer_wise_lr[0]] * len(pos_emb_params))

        if self.cls_token is not None:
            # CLS token params
            cls_token_params = parameter_list(
                named_parameters=self.cls_token.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "cls_token.",
                *args,
                **kwargs,
            )
            param_list.extend(cls_token_params)
            param_lr_list.extend([layer_wise_lr[0]] * len(cls_token_params))

        # transformer related parameters
        for layer_id, transformer_layer in enumerate(self.transformer):
            layer_lr = layer_wise_lr[layer_id + 1]
            transformer_layer_params = parameter_list(
                named_parameters=transformer_layer.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + f"transformer.{layer_id}.",
                *args,
                **kwargs,
            )
            param_list.extend(transformer_layer_params)
            param_lr_list.extend([layer_lr] * len(transformer_layer_params))

        # transformer post-norm params
        post_transformer_norm_params = parameter_list(
            named_parameters=self.post_transformer_norm.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "post_transformer_norm.",
            *args,
            **kwargs,
        )
        param_list.extend(post_transformer_norm_params)
        param_lr_list.extend([layer_wise_lr[-1]] * len(post_transformer_norm_params))

        if self.classifier is not None:
            # classifier parameters
            classifier_params = parameter_list(
                named_parameters=self.classifier.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "classifier.",
                *args,
                **kwargs,
            )
            param_list.extend(classifier_params)
            param_lr_list.extend([1.0] * len(classifier_params))

        if self.simple_fpn is not None:
            # simple FPN parameters
            simple_fpn_params = parameter_list(
                named_parameters=self.simple_fpn.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "simple_fpn.",
                *args,
                **kwargs,
            )
            param_list.extend(simple_fpn_params)
            param_lr_list.extend([1.0] * len(simple_fpn_params))
        return param_list, param_lr_list

    def _build_simple_fpn_layers(
        self, opts, embed_dim: int, norm_layer: str
    ) -> nn.ModuleDict:
        # Helper function to build simple FPN
        layer_l2 = nn.Sequential(
            TransposeConvLayer2d(
                opts,
                in_channels=embed_dim,
                out_channels=embed_dim // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=1,
                use_norm=True,
                use_act=True,
                norm_layer_name=norm_layer,
            ),
            TransposeConvLayer2d(
                opts,
                in_channels=embed_dim // 2,
                out_channels=embed_dim // 4,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=1,
                use_norm=False,
                use_act=False,
                bias=True,
            ),
        )

        self.model_conf_dict["layer2"]["out"] = embed_dim // 4

        layer_l3 = TransposeConvLayer2d(
            opts,
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.model_conf_dict["layer3"]["out"] = embed_dim // 2

        layer_l4 = Identity()
        layer_l5 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        simple_fpn_layers = nn.ModuleDict(
            {
                "out_l2": layer_l2,
                "out_l3": layer_l3,
                "out_l4": layer_l4,
                "out_l5": layer_l5,
            }
        )

        return simple_fpn_layers

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--model.classification.vit.mode",
            type=str,
            default="base",
            help="ViT mode. Default is base.",
        )
        group.add_argument(
            "--model.classification.vit.dropout",
            type=float,
            default=0.0,
            help="Dropout in ViT layers. Defaults to 0.0.",
        )

        group.add_argument(
            "--model.classification.vit.stochastic-dropout",
            type=float,
            default=0.0,
            help="Stochastic Dropout in Transformer layers. Defaults to 0.0.",
        )

        group.add_argument(
            "--model.classification.vit.norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in ViT. Defaults to LayerNorm.",
        )

        group.add_argument(
            "--model.classification.vit.sinusoidal-pos-emb",
            action="store_true",
            default=False,
            help="Use sinusoidal instead of learnable positional encoding. Defaults to False.",
        )
        group.add_argument(
            "--model.classification.vit.no-cls-token",
            action="store_true",
            default=False,
            help="Do not use classification token. Defaults to False.",
        )
        group.add_argument(
            "--model.classification.vit.use-pytorch-mha",
            action="store_true",
            default=False,
            help="Use PyTorch's native multi-head attention. Defaults to False.",
        )

        group.add_argument(
            "--model.classification.vit.use-simple-fpn",
            action="store_true",
            default=False,
            help="Add simple FPN for down-stream tasks. Defaults to False.",
        )

        group.add_argument(
            "--model.classification.vit.checkpoint-segments",
            type=int,
            default=4,
            help="Number of checkpoint segments. Only used when --model.classification.gradient-checkpointing "
            "is enabled. Defaults to 4.",
        )

        return parser

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # input is of shape [Batch, in_channels, height, width]. in_channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]

        # [Batch, in_channels, height, width] --> [Batch, emb_dim, num_patches_height, num_patches_width]
        patch_emb = self.patch_emb(x)
        n_h, n_w = patch_emb.shape[-2:]

        # [Batch, emb_dim, num_patches_height, num_patches_width] --> [Batch, emb_dim, num_patches]
        patch_emb = patch_emb.flatten(2)
        # [Batch, emb_dim, num_patches] --> [Batch, num_patches, emb_dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        n_patches = patch_emb.shape[1]
        # we resize the positional encodings dynamically.
        pos_emb = self.pos_embed(n_patches).to(patch_emb.dtype)

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, emb_dim] --> [Batch, 1, emb_dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concat([Batch, 1, emb_dim], [Batch, num_patches, emb_dim]) --> [Batch, num_patches + 1, emb_dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        # dropout
        patch_emb = self.emb_dropout(patch_emb)
        return patch_emb, (n_h, n_w)

    def _features_from_transformer(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tuple[int, int]]:
        # this function extract patch embeddings and then apply transformer module to learn
        # inter-patch representations

        # [B, N, C] --> [N, B, embed_dim], where B is batch size, N is number of tokens,
        # and embed_dim is feature dim
        x, (n_h, n_w) = self.extract_patch_embeddings(x)

        if self.use_pytorch_mha:
            # For PyTorch MHA, we need sequence first.
            # For our custom MHA implementation, batch is the first
            x = x.transpose(0, 1)

        if self.training and self.gradient_checkpointing:
            # we use sequential checkpoint function, which divides the model into chunks and checkpoints each segment
            # This maybe useful when dealing with large models

            # Right now, gradient checkpoint function does not like kwargs. Therefore, we use default MHA implementation
            # over the PyTorch's fused implementation.
            # Note that default MHA implementation is batch-first, while pytorch implementation is sequence-first.
            x = gradient_checkpoint_fn(self.transformer, self.checkpoint_segments, x)
        else:
            for layer in self.transformer:
                x = layer(x, use_pytorch_mha=self.use_pytorch_mha)
        x = self.post_transformer_norm(x)

        if self.use_pytorch_mha:
            # [N, B, C] --> [B, N, C]
            x = x.transpose(0, 1)

        return x, (n_h, n_w)

    def extract_features(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # The extract_features function for ViT returns two outputs: (1) embedding corresponding to CLS token
        # and (2) image embeddings of the shape [B, C, h//o, w//o], where the value of o is typically 16.
        return_image_embeddings = kwargs.get("return_image_embeddings", False)

        # [B, C, H, W] --> [B, N + 1, embed_dim] or [B, N, embed_dim]
        # here, B is batch size, C is input channels
        # H and W are input height and width
        # N is the number of pixels (or tokens) after processing input with conv stem and reshaping
        # We add +1 for cls token (if applicable)
        # embed_dim --> embedding dimension
        x, (n_h, n_w) = self._features_from_transformer(x, *args, **kwargs)

        if self.cls_token is not None:
            # [B, N + 1, embed_dim] --> [B, embed_dim], [B, N, embed_dim]
            cls_embedding, image_embedding = torch.split(
                x, split_size_or_sections=[1, x.shape[1] - 1], dim=1
            )
            cls_embedding = cls_embedding.squeeze(1)
        else:
            # [B, N, embed_dim] -> [B, embed_dim]
            cls_embedding = torch.mean(x, dim=1)
            # [B, N, embed_dim]
            image_embedding = x

        if return_image_embeddings:
            # reshape image embedding to 4-D tensor
            # [B, N, C] --> [B, C, N]
            image_embedding = image_embedding.transpose(1, 2).contiguous()
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], -1, n_h, n_w
            )

            return cls_embedding, image_embedding
        else:
            return cls_embedding, None

    def forward_classifier(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        cls_embedding, image_embedding = self.extract_features(x, *args, **kwargs)
        # classify based on CLS token
        cls_embedding = self.classifier(cls_embedding)
        return cls_embedding, image_embedding

    def forward(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        # In ViT model, we can return either classifier embeddings (logits) or image embeddings or both.
        # To return the image embeddings, we need to set keyword argument (return_image_embeddings) as True.

        if (
            kwargs.get("return_image_embeddings", False)
            or self.neural_augmentor is not None
        ):
            out_dict = {"augmented_tensor": None}
            if self.training and self.neural_augmentor is not None:
                # neural augmentor is applied during training  only
                x = self.neural_augmentor(x)
                out_dict.update({"augmented_tensor": x})
            prediction, image_embedding = self.forward_classifier(x, *args, **kwargs)
            out_dict.update({"logits": prediction})
            if image_embedding is not None:
                out_dict.update({"image_embeddings": image_embedding})
            return out_dict
        else:
            prediction, _ = self.forward_classifier(x, *args, **kwargs)
            return prediction

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        # this function is often used in down-stream applications (especially in segmentation and detection)
        if self.cls_token:
            logger.error("Please disable cls token for down-stream tasks")

        out_dict = {}
        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        cls_emb, x = self.extract_features(x, return_image_embeddings=True)
        out_dict["cls_embedding"] = cls_emb

        if self.simple_fpn is not None:
            # build simple FPN, as suggested in https://arxiv.org/abs/2203.16527
            for k, extra_layer in self.simple_fpn.items():
                out_dict[k] = extra_layer(x)
        else:
            # ViT does not have hierarchical structure by default.
            # Therefore, we set first four levels to None
            out_dict["out_l1"] = None
            out_dict["out_l2"] = None
            out_dict["out_l3"] = None
            out_dict["out_l4"] = None
            if use_l5_exp:
                out_dict["out_l5"] = None
                out_dict["out_l5_exp"] = x
            else:
                out_dict["out_l5"] = x
                out_dict["out_l5_exp"] = None
        return out_dict

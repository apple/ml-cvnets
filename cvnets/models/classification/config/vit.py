#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict

from utils import logger


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.vit.mode", "tiny")
    if mode is None:
        logger.error("Please specify mode")

    mode = mode.lower()
    dropout = getattr(opts, "model.classification.vit.dropout", 0.0)

    vit_config = dict()
    if mode == "tiny":
        vit_config = {
            "patch_size": 16,
            "embed_dim": 192,
            "n_transformer_layers": 12,
            "n_attn_heads": 3,
            "ffn_dim": 192 * 4,
            "norm_layer": "layer_norm",
            "pos_emb_drop_p": 0.1,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "small":
        vit_config = {
            "patch_size": 16,
            "embed_dim": 384,
            "n_transformer_layers": 12,
            "n_attn_heads": 6,
            "ffn_dim": 384 * 4,
            "norm_layer": "layer_norm",
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    else:
        logger.error("Not supported")
    return vit_config

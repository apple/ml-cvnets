#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
from utils import logger
from typing import Dict
import argparse

from .base_seg_head import BaseSegHead

SEG_HEAD_REGISTRY = {}


def register_segmentation_head(name):
    def register_model_class(cls):
        if name in SEG_HEAD_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(cls, BaseSegHead):
            raise ValueError(
                "Model ({}: {}) must extend BaseSegHead".format(name, cls.__name__)
            )

        SEG_HEAD_REGISTRY[name] = cls
        return cls

    return register_model_class


def build_segmentation_head(opts, enc_conf: Dict, use_l5_exp: bool = False):
    seg_model_name = getattr(opts, "model.segmentation.seg_head", "lr_aspp")
    seg_head = None
    if seg_model_name in SEG_HEAD_REGISTRY:
        seg_head = SEG_HEAD_REGISTRY[seg_model_name](opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)
    else:
        supported_heads = list(SEG_HEAD_REGISTRY.keys())
        supp_model_str = "Supported segmentation heads are:"
        for i, m_name in enumerate(supported_heads):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_model_str)

    return seg_head


def arguments_segmentation_head(parser: argparse.ArgumentParser):
    # add segmentation specific arguments
    for k, v in SEG_HEAD_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


# automatically import the models
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("cvnets.models.segmentation.heads." + model_name)
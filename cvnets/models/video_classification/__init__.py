#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

from utils import logger
from utils.common_utils import check_frozen_norm_layer
from utils.ddp_utils import is_master, is_start_rank_node
from utils.download_utils import get_local_path
from common import SUPPORTED_VIDEO_CLIP_VOTING_FN

from .base_cls import BaseVideoEncoder
from ...misc.common import load_pretrained_model

CLS_MODEL_REGISTRY = {}


def register_video_cls_models(name):
    def register_model_class(cls):
        if name in CLS_MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(cls, BaseVideoEncoder):
            raise ValueError(
                "Model ({}: {}) must extend BaseEncoder".format(name, cls.__name__)
            )

        CLS_MODEL_REGISTRY[name] = cls
        return cls

    return register_model_class


def build_video_classification_model(opts, *args, **kwargs):
    model_name = getattr(opts, "model.video_classification.name", None)
    model = None
    is_master_node = is_master(opts)
    if model_name in CLS_MODEL_REGISTRY:
        cls_act_fn = getattr(opts, "model.video_classification.activation.name", None)
        if cls_act_fn is not None:
            # Override the general activation arguments
            gen_act_fn = getattr(opts, "model.activation.name", "relu")
            gen_act_inplace = getattr(opts, "model.activation.inplace", False)
            gen_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

            setattr(opts, "model.activation.name", cls_act_fn)
            setattr(
                opts,
                "model.activation.inplace",
                getattr(opts, "model.video_classification.activation.inplace", False),
            )
            setattr(
                opts,
                "model.activation.neg_slope",
                getattr(opts, "model.video_classification.activation.neg_slope", 0.1),
            )

            model = CLS_MODEL_REGISTRY[model_name](opts, *args, **kwargs)

            # Reset activation args
            setattr(opts, "model.activation.name", gen_act_fn)
            setattr(opts, "model.activation.inplace", gen_act_inplace)
            setattr(opts, "model.activation.neg_slope", gen_act_neg_slope)

            if is_master_node:
                logger.log(
                    "Overridden the general activation arguments with classification network specific arguments"
                )
        else:
            model = CLS_MODEL_REGISTRY[model_name](opts, *args, **kwargs)
    else:
        supported_models = list(CLS_MODEL_REGISTRY.keys())
        supp_model_str = "Supported models are:"
        for i, m_name in enumerate(supported_models):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))

        if is_master_node:
            logger.error(supp_model_str)

    pretrained = getattr(opts, "model.video_classification.pretrained", None)
    if pretrained is not None:
        pretrained = get_local_path(opts, path=pretrained)
        model = load_pretrained_model(
            model=model, wt_loc=pretrained, is_master_node=is_start_rank_node(opts)
        )

    freeze_norm_layers = getattr(
        opts, "model.video_classification.freeze_batch_norm", False
    )
    if freeze_norm_layers:
        model.freeze_norm_layers()
        frozen_state, count_norm = check_frozen_norm_layer(model)
        if count_norm > 0 and frozen_state and is_master_node:
            logger.error(
                "Something is wrong while freezing normalization layers. Please check"
            )

        if is_master_node:
            logger.log("Normalization layers are frozen")
    return model


def std_video_cls_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Video Classification arguments",
        description="Video Classification arguments",
    )
    group.add_argument(
        "--model.video-classification.classifier-dropout",
        type=float,
        default=0.0,
        help="Dropout rate in classifier",
    )

    group.add_argument(
        "--model.video-classification.name",
        type=str,
        default="mobilevit",
        help="Model name",
    )
    group.add_argument(
        "--model.video-classification.n-classes",
        type=int,
        default=1000,
        help="Number of classes in the dataset",
    )
    group.add_argument(
        "--model.video-classification.pretrained",
        type=str,
        default=None,
        help="Path of the pretrained backbone",
    )
    group.add_argument(
        "--model.video-classification.freeze-batch-norm",
        action="store_true",
        help="Freeze batch norm layers",
    )

    group.add_argument(
        "--model.video-classification.activation.name",
        default=None,
        type=str,
        help="Non-linear function type",
    )
    group.add_argument(
        "--model.video-classification.activation.inplace",
        action="store_true",
        help="Inplace non-linear functions",
    )
    group.add_argument(
        "--model.video-classification.activation.neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu",
    )
    group.add_argument(
        "--model.video-classification.clip-out-voting-fn",
        type=str,
        default="sum",
        choices=SUPPORTED_VIDEO_CLIP_VOTING_FN,
        help="How to fuse the outputs of different clips in a video",
    )

    group.add_argument(
        "--model.video-classification.inference-mode",
        action="store_true",
        help="Inference mode",
    )

    return parser


def arguments_video_classification(parser: argparse.ArgumentParser):
    parser = std_video_cls_model_args(parser=parser)

    # add classification specific arguments
    for k, v in CLS_MODEL_REGISTRY.items():
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
        module = importlib.import_module(
            "cvnets.models.video_classification." + model_name
        )

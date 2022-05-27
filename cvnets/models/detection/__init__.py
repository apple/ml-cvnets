#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from .base_detection import BaseDetection
import os
import importlib
import argparse

from utils.download_utils import get_local_path
from utils import logger
from utils.ddp_utils import is_master, is_start_rank_node
from utils.common_utils import check_frozen_norm_layer

from ...misc.common import load_pretrained_model
from ...models.classification import build_classification_model


DETECT_MODEL_REGISTRY = {}


def register_detection_models(name):
    def register_model_class(cls):
        if name in DETECT_MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(cls, BaseDetection):
            raise ValueError(
                "Model ({}: {}) must extend BaseDetection".format(name, cls.__name__)
            )

        DETECT_MODEL_REGISTRY[name] = cls
        return cls

    return register_model_class


def build_detection_model(opts):
    seg_model_name = getattr(opts, "model.detection.name", None)
    model = None
    is_master_node = is_master(opts)
    if seg_model_name in DETECT_MODEL_REGISTRY:
        output_stride = getattr(opts, "model.detection.output_stride", None)
        encoder = build_classification_model(opts=opts, output_stride=output_stride)
        model = DETECT_MODEL_REGISTRY[seg_model_name](opts, encoder)
    else:
        supported_models = list(DETECT_MODEL_REGISTRY.keys())
        supp_model_str = "Supported detection models are:"
        for i, m_name in enumerate(supported_models):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        if is_master_node:
            logger.error(supp_model_str)

    pretrained = getattr(opts, "model.detection.pretrained", None)
    if pretrained is not None:
        pretrained = get_local_path(opts, path=pretrained)
        model = load_pretrained_model(
            model=model, wt_loc=pretrained, is_master_node=is_start_rank_node(opts)
        )

    freeze_norm_layers = getattr(opts, "model.detection.freeze_batch_norm", False)
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


def common_detection_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Detection arguments", description="Detection arguments"
    )

    group.add_argument(
        "--model.detection.name", type=str, default=None, help="Model name"
    )
    group.add_argument(
        "--model.detection.n-classes",
        type=int,
        default=80,
        help="Number of classes in the dataset",
    )
    group.add_argument(
        "--model.detection.pretrained",
        type=str,
        default=None,
        help="Path of the pretrained model",
    )
    group.add_argument(
        "--model.detection.output-stride",
        type=int,
        default=None,
        help="Output stride in classification network",
    )
    group.add_argument(
        "--model.detection.replace-stride-with-dilation",
        action="store_true",
        help="Replace stride with dilation",
    )
    group.add_argument(
        "--model.detection.freeze-batch-norm",
        action="store_true",
        help="Freeze batch norm layers",
    )

    return parser


def arguments_detection(parser: argparse.ArgumentParser):
    parser = common_detection_args(parser)

    # add segmentation specific arguments
    for k, v in DETECT_MODEL_REGISTRY.items():
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
        module = importlib.import_module("cvnets.models.detection." + model_name)

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
from utils import logger
import argparse
from utils.download_utils import get_local_path
from utils.ddp_utils import is_master
from utils.common_utils import check_frozen_norm_layer

from .. import register_tasks, register_task_arguments
from .base_seg import BaseSegmentation
from ...misc.common import load_pretrained_model
from ..classification import build_classification_model

SEG_MODEL_REGISTRY = {}


def register_segmentation_models(name):
    def register_model_class(cls):
        if name in SEG_MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        if not issubclass(cls, BaseSegmentation):
            raise ValueError(
                "Model ({}: {}) must extend BaseSegmentation".format(name, cls.__name__)
            )

        SEG_MODEL_REGISTRY[name] = cls
        return cls

    return register_model_class


@register_tasks(name="segmentation")
def build_segmentation_model(opts):
    seg_model_name = getattr(opts, "model.segmentation.name", None)
    model = None
    is_master_node = is_master(opts)
    if seg_model_name in SEG_MODEL_REGISTRY:
        output_stride = getattr(opts, "model.segmentation.output_stride", None)
        encoder = build_classification_model(opts=opts, output_stride=output_stride)

        seg_act_fn = getattr(opts, "model.segmentation.activation.name", None)
        if seg_act_fn is not None:
            # Override the general activation arguments
            gen_act_fn = getattr(opts, "model.activation.name", "relu")
            gen_act_inplace = getattr(opts, "model.activation.inplace", False)
            gen_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

            setattr(opts, "model.activation.name", seg_act_fn)
            setattr(
                opts,
                "model.activation.inplace",
                getattr(opts, "model.segmentation.activation.inplace", False),
            )
            setattr(
                opts,
                "model.activation.neg_slope",
                getattr(opts, "model.segmentation.activation.neg_slope", 0.1),
            )

            model = SEG_MODEL_REGISTRY[seg_model_name](opts, encoder)

            # Reset activation args
            setattr(opts, "model.activation.name", gen_act_fn)
            setattr(opts, "model.activation.inplace", gen_act_inplace)
            setattr(opts, "model.activation.neg_slope", gen_act_neg_slope)
        else:
            model = SEG_MODEL_REGISTRY[seg_model_name](opts, encoder)
    else:
        supported_models = list(SEG_MODEL_REGISTRY.keys())
        if len(supported_models) == 0:
            supported_models = ["none"]
        supp_model_str = "Supported segmentation models are:"
        for i, m_name in enumerate(supported_models):
            supp_model_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
        logger.error(supp_model_str)

    finetune_task = getattr(opts, "model.segmentation.finetune_pretrained_model", False)
    pretrained = getattr(opts, "model.segmentation.pretrained", None)
    if finetune_task:
        n_pretrained_classes = getattr(
            opts, "model.segmentation.n_pretrained_classes", None
        )
        n_classes = getattr(opts, "model.segmentation.n_classes", None)
        assert n_pretrained_classes is not None
        assert n_classes is not None

        # The model structure is the same as pre-trained model now
        model.update_classifier(opts, n_classes=n_pretrained_classes)

        # load the weights
        if pretrained is not None:
            pretrained = get_local_path(opts, path=pretrained)
            model = load_pretrained_model(model=model, wt_loc=pretrained, opts=opts)

        # Now, re-initialize the classification layer
        model.update_classifier(opts, n_classes=n_classes)

    elif pretrained is not None:
        pretrained = get_local_path(opts, path=pretrained)
        model = load_pretrained_model(model=model, wt_loc=pretrained, opts=opts)

    freeze_norm_layers = getattr(opts, "model.segmentation.freeze_batch_norm", False)
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


@register_task_arguments(name="segmentation")
def arguments_segmentation(parser: argparse.ArgumentParser):
    parser = BaseSegmentation.add_arguments(parser)

    # add segmentation specific arguments
    for k, v in SEG_MODEL_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    from cvnets.models.segmentation.heads import arguments_segmentation_head

    parser = arguments_segmentation_head(parser)

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
        module = importlib.import_module("cvnets.models.segmentation." + model_name)

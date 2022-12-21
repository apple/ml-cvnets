#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse


from options.utils import extend_selected_args_with_prefix
from cvnets.misc.common import parameter_list
from cvnets.anchor_generator import arguments_anchor_gen
from cvnets.image_projection_layers import arguments_image_projection_head
from cvnets.layers import arguments_nn_layers
from cvnets.matcher_det import arguments_box_matcher
from cvnets.misc.averaging_utils import arguments_ema, EMA
from cvnets.misc.profiler import module_profile
from cvnets.models import arguments_model, get_model
from cvnets.models.detection.base_detection import DetectionPredTuple
from cvnets.neural_augmentor import arguments_neural_augmentor
from cvnets.text_encoders import arguments_text_encoder


def modeling_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # model arguments
    parser = arguments_model(parser)
    # neural network layer argumetns
    parser = arguments_nn_layers(parser)
    # EMA arguments
    parser = arguments_ema(parser)
    # anchor generator arguments (for object detection)
    parser = arguments_anchor_gen(parser)
    # box matcher arguments (for object detection)
    parser = arguments_box_matcher(parser)
    # text encoder arguments (usually for multi-modal tasks)
    parser = arguments_text_encoder(parser)
    # image projection head arguments (usually for multi-modal tasks)
    parser = arguments_image_projection_head(parser)
    # neural aug arguments
    parser = arguments_neural_augmentor(parser)

    # Add teacher as a prefix to enable distillation tasks
    # keep it as the last entry
    parser = extend_selected_args_with_prefix(
        parser, check_string="--model", add_prefix="--teacher."
    )

    return parser

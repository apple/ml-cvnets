#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from torch import nn

from cvnets.models.classification import build_classification_model
from utils import logger


def build_cls_teacher_from_opts(opts) -> nn.Module:
    """
    Helper function to build a classification teacher model from options
    """
    pretrained_model = getattr(opts, "teacher.model.classification.pretrained", None)
    if not pretrained_model:
        logger.error(
            "For distillation, please specify teacher weights using teacher.model.classification.pretrained"
        )

    opts_dict = vars(opts)
    teacher_dict = {
        # replace teacher with empty string in "teacher.model.*" to get model.*
        key.replace("teacher.", ""): value
        for key, value in opts_dict.items()
        # filter keys related to teacher
        if key.split(".")[0] == "teacher"
    }

    # convert to Namespace
    teacher_opts = argparse.Namespace(**teacher_dict)

    # build teacher model
    return build_classification_model(teacher_opts)

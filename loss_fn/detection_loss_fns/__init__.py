#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import importlib
import os

SUPPORTED_DETECTION_LOSS_FNS = []


def register_detection_loss_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_DETECTION_LOSS_FNS:
            raise ValueError("Cannot register duplicate detection loss function ({})".format(name))
        SUPPORTED_DETECTION_LOSS_FNS.append(name)
        return fn
    return register_fn


# automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("loss_fn.detection_loss_fns." + model_name)


# import these after loading loss_fn names to avoid looping
from loss_fn.detection_loss_fns.ssd_multibox_loss import SSDLoss
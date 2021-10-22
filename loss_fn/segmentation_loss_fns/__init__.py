#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib

SUPPORTED_SEG_LOSS_FNS = []


def register_segmentation_loss_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_SEG_LOSS_FNS:
            raise ValueError("Cannot register duplicate segmentation loss function ({})".format(name))
        SUPPORTED_SEG_LOSS_FNS.append(name)
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
        module = importlib.import_module("loss_fn.segmentation_loss_fns." + model_name)


from loss_fn.segmentation_loss_fns.cross_entropy import SegCrossEntropy
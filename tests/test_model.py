#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys
from pathlib import Path
import pytest

sys.path.append("..")

import torch
from torch import Tensor
from typing import Dict

from cvnets import get_model
from loss_fn import build_loss_fn

from tests.configs import get_config


# We use a batch size of 1 to catch error that may arise due to reshaping operations inside the model
@pytest.mark.parametrize("batch_size", [1, 2])
def test_model(config_file: str, batch_size: int):
    opts = get_config(config_file=config_file)
    setattr(opts, "common.debug_mode", True)

    model = get_model(opts)

    criteria = build_loss_fn(opts)

    inputs = None
    targets = None
    if hasattr(model, "dummy_input_and_label"):
        inputs_and_targets = model.dummy_input_and_label(batch_size)
        inputs = inputs_and_targets["samples"]
        targets = inputs_and_targets["targets"]

    assert inputs is not None, (
        "Input tensor can't be None. This is likely because "
        "{} does not implement dummy_input_and_label function".format(
            model.__class__.__name__
        )
    )
    assert targets is not None, (
        "Label tensor can't be None. This is likely because "
        "{} does not implement dummy_input_and_label function".format(
            model.__class__.__name__
        )
    )

    # if getattr(opts, "common.channels_last", False):
    #    inputs = inputs.to(memory_format=torch.channels_last)
    #    model = model.to(memory_format=torch.channels_last)

    try:
        outputs = model(inputs)

        loss = criteria(
            input_sample=inputs,
            prediction=outputs,
            target=targets,
            epoch=0,
            iterations=0,
        )
        print(loss)

        if isinstance(loss, Tensor):
            loss.backward()
        elif isinstance(loss, Dict):
            loss["total_loss"].backward()
        else:
            raise RuntimeError("The output of criteria should be either Dict or Tensor")

        # If there are unused parameters in gradient computation, print them
        # This may be useful for debugging purposes
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)

    except Exception as e:
        if (
            isinstance(e, ValueError)
            and str(e).find("Expected more than 1 value per channel when training") > -1
            and batch_size == 1
        ):
            # For segmentation models (e.g., PSPNet), we pool the tensor so that they have a spatial size of 1.
            # In such a case, batch norm needs a batch size > 1. Otherwise, we can't compute the statistics, raising
            # ValueError("Expected more than 1 value per channel when training"). If we encounter this error
            # for a batch size of 1, we skip it.
            pytest.skip(str(e))
        else:
            raise e


def pytest_generate_tests(metafunc):
    configs = [
        str(x) for x in Path("config").rglob("**/*.yaml") if "tune" not in str(x)
    ]
    metafunc.parametrize("config_file", configs)


if __name__ == "__main__":
    test_model()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys

sys.path.append("..")

import torch


from options.opts import get_training_arguments
from cvnets import get_model
from loss_fn import build_loss_fn
from utils.tensor_utils import create_rand_tensor
from utils import logger


def test_model(*args, **kwargs):
    opts = get_training_arguments()

    model = get_model(opts)
    loss_fn = build_loss_fn(opts)

    inp = create_rand_tensor(opts)

    if getattr(opts, "common.channels_last", False):
        inp = inp.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

        if not inp.is_contiguous(memory_format=torch.channels_last):
            logger.warning(
                "Unable to convert input to channels_last format. Setting model to contiguous format"
            )
            model = model.to(memory_format=torch.contiguous_format)

    # FLOPs computed using model.profile_model and fvcore can be different because
    # model.profile_model ignore some of the operations (e.g., addition) while
    # fvcore accounts for all operations (e.g., addition)
    model.profile_model(inp)
    model.eval()
    out = model(inp)

    try:
        # compute flops using FVCore also
        from fvcore.nn import FlopCountAnalysis

        flop_analyzer = FlopCountAnalysis(model.eval(), inp)
        flop_analyzer.unsupported_ops_warnings(False)
        flop_analyzer.uncalled_modules_warnings(False)
        total_flops = flop_analyzer.total()

        print(
            "Flops computed using FVCore for an input of size={} are {:>8.3f} G".format(
                list(inp.shape), total_flops / 1e9
            )
        )
    except ModuleNotFoundError:
        pass

    try:
        n_classes = out.shape[1]

        pred = torch.argmax(out, dim=1)
        targets = torch.randint(0, n_classes, size=pred.shape)
        loss = loss_fn(None, out, targets)
        loss.backward()

        print(model)
        print(loss_fn)
        print("Random Input : {}".format(inp.shape))
        print("Random Target: {}".format(targets.shape))
        print("Random Output: {}".format(out.shape))
    except:
        print(model)


if __name__ == "__main__":
    test_model()

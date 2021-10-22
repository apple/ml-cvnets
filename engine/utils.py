#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from utils import logger
import torch
from typing import Optional
import gc

from utils.ddp_utils import is_master
from utils.tensor_utils import create_rand_tensor


def print_summary(opts, model, criteria: Optional = None, optimizer: Optional = None, scheduler: Optional = None):
    if is_master(opts):
        logger.log(logger.color_text('Model'))
        print(model)
        dev = getattr(opts, "dev.device", torch.device("cpu"))
        try:
            inp_tensor = create_rand_tensor(opts, device=dev)

            if hasattr(model, 'module'):
                model.module.profile_model(inp_tensor)
            else:
                model.profile_model(inp_tensor)
            del inp_tensor
        except Exception as e:
            pass

        if criteria is not None:
            # print criteria
            logger.log(logger.color_text('Loss function'))
            print('{}'.format(criteria))

        if optimizer is not None:
            logger.log(logger.color_text('Optimizer'))
            print('{}'.format(optimizer))

        if scheduler is not None:
            logger.log(logger.color_text('Learning rate scheduler'))
            print('{}'.format(scheduler))

        gc.collect()

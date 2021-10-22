#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor


def module_profile(module, x: Tensor) -> (Tensor, float, float):
    # Note: Module profiling is for reference only and may contain errors.
    # Relies on user to implement these functions accurately.

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)

    return x, n_params, n_macs


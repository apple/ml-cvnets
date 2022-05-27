#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Tuple


def module_profile(module, x: Tensor, *args, **kwargs) -> Tuple[Tensor, float, float]:
    """
    Helper function to profile a module.

    .. note::
        Module profiling is for reference only and may contain errors as it solely relies on user implementation to
        compute theoretical FLOPs
    """

    if isinstance(module, nn.Sequential):
        n_macs = n_params = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profile_module(x)
                n_macs += l_macs
                n_params += l_p
            except Exception as e:
                print(e, l)
                pass
    else:
        x, n_params, n_macs = module.profile_module(x)
    return x, n_params, n_macs

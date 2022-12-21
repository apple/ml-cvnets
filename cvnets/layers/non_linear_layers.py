#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn
from typing import Optional

from .activation import build_activation_layer


def get_activation_fn(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    return build_activation_layer(
        act_type=act_type,
        num_parameters=num_parameters,
        negative_slope=negative_slope,
        inplace=inplace,
        *args,
        **kwargs
    )

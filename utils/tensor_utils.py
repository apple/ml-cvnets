#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist
from typing import Union, Optional, Tuple

from common import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_CHANNELS


def tensor_size_from_opts(opts) -> Tuple[int, int]:
    try:
        sampler_name = getattr(opts, "sampler.name", "variable_batch_sampler").lower()
        if sampler_name.find("var") > -1:
            im_w = getattr(opts, "sampler.vbs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.vbs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
        else:
            im_w = getattr(opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
    except Exception as e:
        im_w = im_h = 256
    return im_h, im_w


def create_rand_tensor(opts, device: Optional[str] = "cpu") -> Tensor:
    im_h, im_w = tensor_size_from_opts(opts=opts)
    inp_tensor = torch.randint(low=0, high=255, size=(1, DEFAULT_IMAGE_CHANNELS, im_h, im_w), device=device)
    inp_tensor = inp_tensor.float().div(255.0)
    return inp_tensor


def reduce_tensor(inp_tensor: torch.Tensor) -> torch.Tensor:
    size = float(dist.get_world_size())
    inp_tensor_clone = inp_tensor.clone()
    dist.barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def tensor_to_python_float(inp_tensor: Union[int, float, torch.Tensor],
                           is_distributed: bool) -> Union[int, float, np.ndarray]:
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        inp_tensor = reduce_tensor(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, 'item'):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError("The data type is not supported yet in tensor_to_python_float function")


def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np

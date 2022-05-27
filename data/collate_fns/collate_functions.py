#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from typing import List, Dict

from utils import logger

from . import register_collate_fn


@register_collate_fn(name="default_collate_fn")
def default_collate_fn(batch: List[Dict], opts):
    """Default collate function"""
    batch_size = len(batch)

    keys = list(batch[0].keys())

    new_batch = {k: [] for k in keys}
    for b in range(batch_size):
        for k in keys:
            new_batch[k].append(batch[b][k])

    # stack the keys
    for k in keys:
        batch_elements = new_batch.pop(k)

        if isinstance(batch_elements[0], (int, float)):
            # list of ints or floats
            batch_elements = torch.as_tensor(batch_elements)
        else:
            # stack tensors (including 0-dimensional)
            try:
                batch_elements = torch.stack(batch_elements, dim=0).contiguous()
            except Exception as e:
                logger.error("Unable to stack the tensors. Error: {}".format(e))

        if k == "image" and getattr(opts, "common.channels_last", False):
            batch_elements = batch_elements.to(memory_format=torch.channels_last)

        new_batch[k] = batch_elements

    return new_batch

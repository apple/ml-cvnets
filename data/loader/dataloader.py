#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from typing import Optional, Union, List
from torch.utils.data import DataLoader

from ..sampler.base_sampler import BaseSamplerDP, BaseSamplerDDP
from ..datasets.dataset_base import BaseImageDataset


class CVNetsDataLoader(DataLoader):
    """This class extends PyTorch's Dataloader"""

    def __init__(
        self,
        dataset: BaseImageDataset,
        batch_size: int,
        batch_sampler: Union[BaseSamplerDP, BaseSamplerDDP],
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        collate_fn: Optional = None,
        prefetch_factor: Optional[int] = 2,
        *args,
        **kwargs
    ):
        super(CVNetsDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
        )

    def update_indices(self, new_indices: List, *args, **kwargs):
        """Update indices in the dataset class"""
        if hasattr(self.batch_sampler, "img_indices") and hasattr(
            self.batch_sampler, "update_indices"
        ):
            self.batch_sampler.update_indices(new_indices)

    def samples_in_dataset(self):
        """Number of samples in the dataset"""
        return len(self.batch_sampler.img_indices)

    def get_sample_indices(self) -> List:
        """Sample IDs"""
        return self.batch_sampler.img_indices

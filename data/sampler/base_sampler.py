#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch.utils.data.sampler import Sampler
from typing import Optional
import torch.distributed as dist
import math
import argparse


class BaseSamplerDP(Sampler):
    """
        Base class for DataParallel Sampler
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        n_gpus: int = max(1, torch.cuda.device_count())
        batch_size_gpu0: int = getattr(opts, "dataset.train_batch_size0", 32) if is_training \
            else getattr(opts, "dataset.val_batch_size0", 32)

        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus

        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[:(total_size - n_data_samples)]
        assert total_size == len(indexes)

        self.img_indices = indexes
        self.n_samples = total_size
        self.batch_size_gpu0 = batch_size_gpu0
        self.n_gpus = n_gpus
        self.shuffle = True if is_training else False
        self.epoch = 0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class BaseSamplerDDP(Sampler):
    """
        Base class for DistributedDataParallel sampler
    """
    def __init__(self, opts, n_data_samples: int, is_training: Optional[bool] = False):
        """

        :param opts: arguments
        :param n_data_samples: number of data samples in the dataset
        :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        batch_size_gpu0: int = getattr(opts, "dataset.train_batch_size0", 32) if is_training \
            else getattr(opts, "dataset.val_batch_size0", 32)

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        num_replicas = dist.get_world_size()
        rank = dist.get_rank()

        num_samples_per_replica = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas

        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[:(total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.shuffle = True if is_training else False
        self.epoch = 0
        self.rank = rank
        self.batch_size_gpu0 = batch_size_gpu0
        self.num_replicas = num_replicas

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.n_samples_per_replica

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
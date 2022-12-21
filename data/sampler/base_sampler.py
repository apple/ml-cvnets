#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch.utils.data.sampler import Sampler
from typing import Optional
import torch.distributed as dist
import math
import argparse
import copy
import numpy as np
import random


class BaseSamplerDP(Sampler):
    """
    Base class for DataParallel Sampler

    Args:
        opts: command line argument
        n_data_samples (int): Number of samples in the dataset
        is_training (Optional[bool]): Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        n_gpus: int = max(1, torch.cuda.device_count())
        batch_size_gpu0: int = (
            getattr(opts, "dataset.train_batch_size0", 32)
            if is_training
            else getattr(opts, "dataset.val_batch_size0", 32)
        )

        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus

        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[: (total_size - n_data_samples)]
        assert total_size == len(indexes)

        self.img_indices = indexes
        self.n_samples = total_size
        self.batch_size_gpu0 = batch_size_gpu0
        self.n_gpus = n_gpus
        self.shuffle = True if is_training else False
        self.epoch = 0

        self.num_repeats = getattr(opts, "sampler.num_repeats", 1) if is_training else 1
        self.trunc_rep_aug = getattr(
            opts, "sampler.truncated_repeat_aug_sampler", False
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def extra_repr(self):
        extra_repr_str = "\n\t num_repeat={}" "\n\t trunc_rep_aug={}".format(
            self.num_repeats, self.trunc_rep_aug
        )
        return extra_repr_str

    def get_indices(self):
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(img_indices)

            if self.num_repeats > 1:
                # Apply repeated augmentation
                """Assume that we have [0, 1, 2, 3] samples. With repeated augmentation,
                we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                samples [0, 0, 0, 1]. Note that we do shuffle at the beginning, so samples are not the
                same at every iteration.
                """
                n_samples_before_repeat = len(img_indices)
                img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                img_indices = list(img_indices)
                if self.trunc_rep_aug:
                    img_indices = img_indices[:n_samples_before_repeat]
        return img_indices

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.img_indices) * (1 if self.trunc_rep_aug else self.num_repeats)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

    def update_indices(self, new_indices):
        self.img_indices = new_indices

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class BaseSamplerDDP(Sampler):
    """
    Base class for DistributedDataParallel Sampler

    Args:
        opts: command line argument
        n_data_samples (int): Number of samples in the dataset
        is_training (Optional[bool]): Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        batch_size_gpu0: int = (
            getattr(opts, "dataset.train_batch_size0", 32)
            if is_training
            else getattr(opts, "dataset.val_batch_size0", 32)
        )

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        gpus_node_i = max(1, torch.cuda.device_count())

        num_samples_per_replica = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas

        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[: (total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.shuffle = True if is_training else False
        self.epoch = 0
        self.rank = rank
        self.batch_size_gpu0 = batch_size_gpu0
        self.num_replicas = num_replicas
        self.skip_sample_indices = []
        self.node_id = rank // gpus_node_i

        self.num_nodes = max(1, num_replicas // gpus_node_i)
        self.local_rank = rank % gpus_node_i
        self.num_gpus_node_i = gpus_node_i

        self.sharding = (
            getattr(opts, "sampler.use_shards", False) if is_training else False
        )
        self.num_repeats = getattr(opts, "sampler.num_repeats", 1) if is_training else 1
        self.trunc_rep_aug = (
            getattr(opts, "sampler.truncated_repeat_aug_sampler", False)
            if self.num_repeats
            else False
        )
        self.n_samples_per_replica = num_samples_per_replica * (
            1 if self.trunc_rep_aug else self.num_repeats
        )
        self.disable_shuffle_sharding = getattr(
            opts, "sampler.disable_shuffle_sharding", False
        )

    def extra_repr(self):
        extra_repr_str = (
            "\n\t num_repeat={}"
            "\n\t trunc_rep_aug={}"
            "\n\t sharding={}"
            "\n\t disable_shuffle_sharding={}".format(
                self.num_repeats,
                self.trunc_rep_aug,
                self.sharding,
                self.disable_shuffle_sharding,
            )
        )
        return extra_repr_str

    def get_indices_rank_i(self):
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)

            if self.sharding:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive first 4 samples and node 1 will receive last 4 samples.

                note:
                    This strategy is useful when dataset is large and we want to process subset of dataset on each node.
                """

                # compute number pf samples per node.
                # Each node may have multiple GPUs
                # Node id = rank // num_gpus_per_rank
                samples_per_node = int(math.ceil(len(img_indices) / self.num_nodes))
                indices_node_i = img_indices[
                    self.node_id
                    * samples_per_node : (self.node_id + 1)
                    * samples_per_node
                ]

                # Ensure that each node has equal number of samples
                if len(indices_node_i) < samples_per_node:
                    indices_node_i += indices_node_i[
                        : (samples_per_node - len(indices_node_i))
                    ]

                # Note: For extremely large datasets, we may want to disable shuffling for efficient data loading
                if not self.disable_shuffle_sharding:
                    # shuffle the indices within a node.
                    random.shuffle(indices_node_i)

                if self.num_repeats > 1:
                    """Assume that we have [0, 1, 2, 3] samples in rank_i. With repeated augmentation,
                    we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                    samples [0, 0, 0, 1]. Note shuffling at the beginning
                    """
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(indices_node_i)
                    indices_node_i = np.repeat(indices_node_i, repeats=self.num_repeats)
                    indices_node_i = list(indices_node_i)
                    if self.trunc_rep_aug:
                        indices_node_i = indices_node_i[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = indices_node_i[
                    self.local_rank : len(indices_node_i) : self.num_gpus_node_i
                ]
            else:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive [0, 2, 4, 6] and node 1 will receive [1, 3, 4, 7].

                note:
                    This strategy is useful when each data sample is stored independently, and is
                    default in many frameworks
                """
                random.shuffle(img_indices)

                if self.num_repeats > 1:
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(img_indices)
                    img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                    img_indices = list(img_indices)
                    if self.trunc_rep_aug:
                        img_indices = img_indices[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = img_indices[
                    self.rank : len(img_indices) : self.num_replicas
                ]
        else:
            indices_rank_i = img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]
        return indices_rank_i

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return (len(self.img_indices) // self.num_replicas) * (
            1 if self.trunc_rep_aug else self.num_repeats
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

    def update_indices(self, new_indices):
        self.img_indices = new_indices

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

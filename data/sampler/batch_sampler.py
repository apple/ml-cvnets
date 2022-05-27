#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import argparse
from typing import Optional

import numpy as np

from common import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

from . import register_sampler, BaseSamplerDDP, BaseSamplerDP


@register_sampler(name="batch_sampler")
class BatchSampler(BaseSamplerDP):
    """
    Standard Batch Sampler for data parallel

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
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        crop_size_w: int = getattr(
            opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH
        )
        crop_size_h: int = getattr(
            opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT
        )

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.num_repeats = (
            getattr(opts, "sampler.bs.num_repeats", 1) if is_training else 1
        )

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            img_indices = np.repeat(self.img_indices, repeats=self.num_repeats)
            img_indices = list(img_indices)
            random.shuffle(img_indices)
        else:
            img_indices = self.img_indices

        start_index = 0
        batch_size = self.batch_size_gpu0
        n_samples = len(img_indices)
        while start_index < n_samples:

            end_index = min(start_index + batch_size, n_samples)
            batch_ids = img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids
                ]
                yield batch

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += (
            "\n\tbase_im_size=(h={}, w={})"
            "\n\tbase_batch_size={},"
            "\n\tnum_repeats={}".format(
                self.crop_size_h,
                self.crop_size_w,
                self.batch_size_gpu0,
                self.num_repeats,
            )
        )
        repr_str += "\n)"
        return repr_str

    def __len__(self):
        return len(self.img_indices) * self.num_repeats

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Batch sampler", description="Arguments related to Batch sampler"
        )
        group.add_argument(
            "--sampler.bs.crop-size-width",
            default=DEFAULT_IMAGE_WIDTH,
            type=int,
            help="Base crop size (along width) during training",
        )
        group.add_argument(
            "--sampler.bs.crop-size-height",
            default=DEFAULT_IMAGE_HEIGHT,
            type=int,
            help="Base crop size (along height) during training",
        )
        group.add_argument(
            "--sampler.bs.num-repeats",
            type=int,
            default=1,
            help="Repeat each sample x times during an epoch",
        )
        return parser


@register_sampler(name="batch_sampler_ddp")
class BatchSamplerDDP(BaseSamplerDDP):
    """
    Standard Batch Sampler for distributed data parallel

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
        super().__init__(
            opts=opts, n_data_samples=n_data_samples, is_training=is_training
        )
        crop_size_w: int = getattr(
            opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH
        )
        crop_size_h: int = getattr(
            opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT
        )

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.num_repeats = (
            getattr(opts, "sampler.bs.num_repeats", 1) if is_training else 1
        )

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)

            img_indices = np.repeat(self.img_indices, repeats=self.num_repeats)
            img_indices = list(img_indices)
            indices_rank_i = img_indices[
                self.rank : len(img_indices) : self.num_replicas
            ]
            random.shuffle(indices_rank_i)
        else:
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]

        start_index = 0
        batch_size = self.batch_size_gpu0

        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids
                ]
                yield batch

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += (
            "\n\tbase_im_size=(h={}, w={})"
            "\n\tbase_batch_size={}"
            "\n\tnum_repeats={}".format(
                self.crop_size_h,
                self.crop_size_w,
                self.batch_size_gpu0,
                self.num_repeats,
            )
        )
        repr_str += "\n)"
        return repr_str

    def __len__(self):
        return (len(self.img_indices) // self.num_replicas) * self.num_repeats

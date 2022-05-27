#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import argparse
from typing import Optional

from . import register_sampler
from .batch_sampler import BatchSamplerDDP, BatchSampler


@register_sampler(name="video_batch_sampler")
class VideoBatchSampler(BatchSampler):
    """
    Batch sampler for videos

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
        self.default_frames = getattr(opts, "sampler.bs.num_frames_per_clip", 8)

        self.clips_per_video = getattr(opts, "sampler.bs.clips_per_video", 1)

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples:

            end_index = min(start_index + batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (
                        self.crop_size_h,
                        self.crop_size_w,
                        b_id,
                        self.default_frames,
                        self.clips_per_video,
                    )
                    for b_id in batch_ids
                ]
                yield batch

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Batch sampler for videos",
            description="Arguments related to variable batch sampler",
        )
        group.add_argument(
            "--sampler.bs.num-frames-per-clip",
            default=8,
            type=int,
            help="Number of frames per video clip",
        )
        group.add_argument(
            "--sampler.bs.clips-per-video",
            default=1,
            type=int,
            help="Number of clips per video",
        )
        return parser

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n \t n_clips={}\n \tn_frames={}".format(
            self.crop_size_h,
            self.crop_size_w,
            self.batch_size_gpu0,
            self.clips_per_video,
            self.default_frames,
        )
        repr_str += "\n)"
        return repr_str


@register_sampler(name="video_batch_sampler_ddp")
class VideoBatchSamplerDDP(BatchSamplerDDP):
    """
    Batch sampler for videos (DDP)

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
        self.default_frames = getattr(opts, "sampler.bs.num_frames_per_clip", 8)
        self.clips_per_video = getattr(opts, "sampler.bs.clips_per_video", 1)

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]
            random.shuffle(indices_rank_i)
        else:
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples_per_replica:
            end_index = min(start_index + batch_size, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (
                        self.crop_size_h,
                        self.crop_size_w,
                        b_id,
                        self.default_frames,
                        self.clips_per_video,
                    )
                    for b_id in batch_ids
                ]
                yield batch

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n \t base_im_size=(h={}, w={})\n \t base_batch_size={}\n \t n_clips={}\n \tn_frames={}".format(
            self.crop_size_h,
            self.crop_size_w,
            self.batch_size_gpu0,
            self.clips_per_video,
            self.default_frames,
        )
        repr_str += "\n)"
        return repr_str

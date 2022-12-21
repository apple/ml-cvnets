#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import argparse
from typing import Optional

from utils import logger

from .utils import make_video_pairs
from . import register_sampler
from .variable_batch_sampler import VariableBatchSampler, VariableBatchSamplerDDP


@register_sampler(name="video_variable_seq_sampler")
class VideoVariableSeqSampler(VariableBatchSampler):
    """
    Extends `Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for videos

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
        self.default_frames = getattr(opts, "sampler.vbs.num_frames_per_clip", 8)

        self.random_video_clips = (
            getattr(opts, "sampler.vbs.random_video_clips", False)
            if is_training
            else False
        )
        self.min_clips_per_video = getattr(opts, "sampler.vbs.min_clips_per_video", 1)
        self.max_clips_per_video = getattr(opts, "sampler.vbs.max_clips_per_video", 5)
        self.clips_per_video = getattr(opts, "sampler.vbs.clips_per_video", 1)
        if self.min_clips_per_video is None:
            self.min_clips_per_video = 1

        if is_training:
            # override img_batch_tuples
            self.img_batch_tuples = make_video_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                default_frames=self.default_frames,
            )
        else:
            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.default_frames)
            ]

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)

        if self.shuffle:
            n_clips_str = "(min={}, max={})".format(
                self.min_clips_per_video, self.max_clips_per_video
            )
        else:
            n_clips_str = self.clips_per_video

        repr_str += (
            "\n\t base_im_size=(h={}, w={}), "
            "\n\t base_batch_size={} "
            "\n\t scales (Height x Width x N_frames)={} "
            "\n\t scale_inc={} "
            "\n\t min_scale_inc_factor={} "
            "\n\t max_scale_inc_factor={} "
            "\n\t ep_intervals={}"
            "\n\t num_repeat={}"
            "\n\t num_clips={}".format(
                self.crop_size_h,
                self.crop_size_w,
                self.batch_size_gpu0,
                self.img_batch_tuples,
                self.scale_inc,
                self.min_scale_inc_factor,
                self.max_scale_inc_factor,
                self.scale_ep_intervals,
                self.num_repeats,
                n_clips_str,
            )
        )
        repr_str += self.extra_repr()
        repr_str += "\n)"
        return repr_str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Variable sequence sampler for videos",
            description="Arguments related to variable sequence sampler",
        )
        group.add_argument(
            "--sampler.vbs.num-frames-per-clip",
            default=8,
            type=int,
            help="Default frames per video",
        )

        group.add_argument(
            "--sampler.vbs.random-video-clips",
            action="store_true",
            help="Sample number of clips per video randomly during training between min and max values specified using "
            "--dataset.kinetics.min-clips-per-video and --dataset.kinetics.max-clips-per-video arguments "
            "respectively",
        )
        group.add_argument(
            "--sampler.vbs.min-clips-per-video",
            type=int,
            default=1,
            help="Minimum number of clips per video. Used only for training",
        )
        group.add_argument(
            "--sampler.vbs.max-clips-per-video",
            type=int,
            default=5,
            help="Maximum number of clips per video. Used only for training",
        )
        group.add_argument(
            "--sampler.vbs.clips-per-video",
            type=int,
            default=1,
            help="Number of clips per video",
        )
        group.add_argument(
            "--sampler.vbs.min-frames-per-clip",
            type=int,
            default=None,
            help="Minimum number of frames per clip",
        )

        return parser

    def __iter__(self):
        indices = self.get_indices()

        start_index = 0
        indices_len = len(indices)
        while start_index < indices_len:
            if self.random_video_clips:
                # randomly sample number of clips and adjust frames per clip
                n_clips = max(
                    1,
                    random.randint(self.min_clips_per_video, self.max_clips_per_video),
                )
                batch_size = max(
                    self.batch_size_gpu0,
                    self.batch_size_gpu0 * (self.clips_per_video // n_clips),
                )
            else:
                n_clips = self.clips_per_video
                batch_size = self.batch_size_gpu0

            crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)
            end_index = min(start_index + batch_size, indices_len)
            batch_ids = indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += indices[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:

                batch = [
                    (crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
                ]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass


@register_sampler(name="video_variable_seq_sampler_ddp")
class VideoVariableSeqSamplerDDP(VariableBatchSamplerDDP):
    """
    Extends `Variably-size multi-scale batch sampler <https://arxiv.org/abs/2110.02178?context=cs.LG>` for videos

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
        self.default_frames = getattr(opts, "sampler.vbs.num_frames_per_clip", 8)

        self.random_video_clips = (
            getattr(opts, "sampler.vbs.random_video_clips", False)
            if is_training
            else False
        )
        self.min_clips_per_video = getattr(opts, "sampler.vbs.min_clips_per_video", 1)
        self.max_clips_per_video = getattr(opts, "sampler.vbs.max_clips_per_video", 5)
        self.clips_per_video = getattr(opts, "sampler.vbs.clips_per_video", 1)
        if self.min_clips_per_video is None:
            self.min_clips_per_video = 1

        if is_training:
            # override img_batch_tuples
            self.img_batch_tuples = make_video_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                default_frames=self.default_frames,
            )
        else:
            self.img_batch_tuples = [
                (self.crop_size_h, self.crop_size_w, self.default_frames)
            ]

    def __repr__(self):
        repr_str = "{}(".format(self.__class__.__name__)

        if self.shuffle:
            n_clips_str = "(min={}, max={})".format(
                self.min_clips_per_video, self.max_clips_per_video
            )
        else:
            n_clips_str = self.clips_per_video

        repr_str += (
            "\n\t base_im_size=(h={}, w={}), "
            "\n\t base_batch_size={} "
            "\n\t scales (Height x Width x N_frames)={} "
            "\n\t scale_inc={} "
            "\n\t min_scale_inc_factor={} "
            "\n\t max_scale_inc_factor={} "
            "\n\t ep_intervals={}"
            "\n\t num_repeat={}"
            "\n\t num_clips={}".format(
                self.crop_size_h,
                self.crop_size_w,
                self.batch_size_gpu0,
                self.img_batch_tuples,
                self.scale_inc,
                self.min_scale_inc_factor,
                self.max_scale_inc_factor,
                self.scale_ep_intervals,
                self.num_repeats,
                n_clips_str,
            )
        )
        repr_str += self.extra_repr()
        repr_str += "\n)"
        return repr_str

    def __iter__(self):
        indices_rank_i = self.get_indices_rank_i()

        start_index = 0
        n_samples_rank_i = len(indices_rank_i)
        while start_index < n_samples_rank_i:
            if self.random_video_clips:
                # randomly sample number of clips and adjust batch size
                n_clips = max(
                    1,
                    random.randint(self.min_clips_per_video, self.max_clips_per_video),
                )
                batch_size = max(
                    self.batch_size_gpu0,
                    self.batch_size_gpu0 * (self.clips_per_video // n_clips),
                )
            else:
                n_clips = self.clips_per_video
                batch_size = self.batch_size_gpu0

            crop_h, crop_w, n_frames = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, n_samples_rank_i)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[: (batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (crop_h, crop_w, b_id, n_frames, n_clips) for b_id in batch_ids
                ]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        pass

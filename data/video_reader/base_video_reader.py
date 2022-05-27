#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
from typing import Optional, Any, List
import torch
import numpy as np
import av
from torch import Tensor
import random
import argparse
from PIL import Image
from torchvision.transforms import functional as F_vision

from utils import logger
from ..transforms import image_pil as T


class PyAVBaseReader(object):
    """
    PyAv video reader

    Args:
        opts: command line arguments
        is_training (Optional[bool]): Training or validation mode. Default: `False`
    """

    def __init__(self, opts, is_training: Optional[bool] = False, *args, **kwargs):
        super().__init__()
        self.fast_decoding = getattr(opts, "video_reader.fast_video_decoding", False)
        self.frame_stack_format = getattr(
            opts, "video_reader.frame_stack_format", "sequence_first"
        )
        self.stack_frame_dim = 1 if self.frame_stack_format == "channel_first" else 0

        self.frame_transforms = (
            self._frame_transform(opts=opts) if is_training else None
        )
        self.random_erase_transform = (
            self._random_erase_transform(opts=opts) if is_training else None
        )

        self.frame_transforms_str = ""
        if self.frame_transforms is not None:
            self.frame_transforms_str += "\t {}".format(
                self.frame_transforms.__repr__()
            )
        if self.random_erase_transform is not None:
            self.frame_transforms_str += "\t {}".format(
                self.random_erase_transform.__repr__()
            )

        self.num_frames_cache = dict()

    @staticmethod
    def _frame_transform(opts):
        auto_augment = getattr(opts, "image_augmentation.auto_augment.enable", False)
        rand_augment = getattr(opts, "image_augmentation.rand_augment.enable", False)

        if auto_augment and rand_augment:
            logger.warning(
                "AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"
            )
        elif auto_augment:
            return T.AutoAugment(opts=opts)
        elif rand_augment:
            return T.RandAugment(opts=opts)
        return None

    @staticmethod
    def _random_erase_transform(opts):
        random_erase = getattr(opts, " image_augmentation.random_erase.enable", False)
        if random_erase:
            return T.RandomErasing(opts=opts)
        return None

    def __repr__(self):
        return "{}(\n\tfast_decoding={}\n\tframe_stack_format={}\n)".format(
            self.__class__.__name__, self.fast_decoding, self.frame_stack_format
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def check_video(self, filename: str) -> bool:
        try:
            # Adapted from basic demo: https://pyav.org/docs/stable/#basic-demo
            with av.open(filename) as container:
                # Decode the first video channel.
                for frame in container.decode(video=0):
                    frame_idx = frame.index
                    break
                return True
        except Exception as e:
            return False

    def read_video(self, filename: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    def num_frames(self, filename: str) -> int:
        if filename in self.num_frames_cache:
            return self.num_frames_cache[filename]
        else:
            total_frames = 0
            with av.open(filename) as container:
                total_frames = container.streams.video[0].frames
            self.num_frames_cache[filename] = total_frames
            return total_frames

    def frame_to_tensor(self, frame):
        frame_np = frame.to_ndarray(format="rgb24")
        if self.frame_transforms is not None:
            #
            frame_pil = Image.fromarray(frame_np)
            frame_pil = self.frame_transforms({"image": frame_pil})["image"]
            frame_np = np.array(frame_pil)

        frame_np = frame_np.transpose(2, 0, 1)
        frame_np = np.ascontiguousarray(frame_np)
        # [C, H, W]
        frame_torch = torch.from_numpy(frame_np)

        # normalize the frame between 0 and 1
        frame_torch = frame_torch.div(255.0)

        # apply random erase transform
        if self.random_erase_transform is not None:
            frame_torch = self.random_erase_transform({"image": frame_torch})["image"]

        return frame_torch

    @staticmethod
    def random_sampling(
        desired_frames: int, total_frames: int, n_clips: int, *args, **kwargs
    ) -> List:
        # divide the video into K clips
        try:
            interval = (
                desired_frames if total_frames >= desired_frames * (n_clips + 1) else 1
            )
            # The range of start Id is between [0, total_frames - n_desired_frames]
            temp = max(0, min(total_frames - desired_frames, total_frames))
            start_ids = sorted(
                random.sample(population=range(0, temp, interval), k=n_clips)
            )
            # 30 frames and 120 frames in 1s and 4s videos @ 30 FPS, respectively
            # The end_id is randomly selected between start_id + 30 and start_id + 120
            end_ids = [
                min(
                    max(s_id + random.randint(30, 120), s_id + desired_frames),
                    total_frames - 1,
                )
                for s_id in start_ids
            ]
        except:
            # fall back to uniform
            video_clip_ids = np.linspace(
                0, total_frames - 1, n_clips + 1, dtype=int
            ).tolist()

            start_ids = video_clip_ids[:-1]
            end_ids = video_clip_ids[1:]

        frame_ids = []
        for start_idx, end_idx in zip(start_ids, end_ids):
            try:
                clip_frame_ids = sorted(
                    random.sample(
                        population=range(start_idx, end_idx), k=desired_frames
                    )
                )
            except:
                # sample with repetition
                clip_frame_ids = np.linspace(
                    start=start_idx, stop=end_idx - 1, num=desired_frames, dtype=int
                ).tolist()
            frame_ids.extend(clip_frame_ids)
        return frame_ids

    @staticmethod
    def uniform_sampling(
        desired_frames: int, total_frames: int, n_clips: int, *args, **kwargs
    ):
        video_clip_ids = np.linspace(
            0, total_frames - 1, n_clips + 1, dtype=int
        ).tolist()
        start_ids = video_clip_ids[:-1]
        end_ids = video_clip_ids[1:]

        frame_ids = []
        for start_idx, end_idx in zip(start_ids, end_ids):
            clip_frame_ids = np.linspace(
                start=start_idx, stop=end_idx - 1, num=desired_frames, dtype=int
            ).tolist()
            frame_ids.extend(clip_frame_ids)
        return frame_ids

    def convert_to_clips(self, video: torch.Tensor, n_clips: int):
        # video is [N, C, H, W] or [C, N, H, W]
        video_clips = torch.chunk(video, chunks=n_clips, dim=self.stack_frame_dim)
        video_clips = torch.stack(video_clips, dim=0)
        # video_clips is [T, n, C, H, W] or [T, C, n, H, W]
        return video_clips

    def process_video(
        self,
        vid_filename: str,
        n_frames_per_clip: Optional[int] = -1,
        clips_per_video: Optional[int] = 1,
        video_transform_fn: Optional = None,
        is_training: Optional[bool] = False,
    ):
        raise NotImplementedError

    def dummy_video(
        self, clips_per_video: int, n_frames_to_sample: int, height: int, width: int
    ):

        # [K, C, N, H, W] or # [K, N, C, H, W]
        # K --> number of clips, C --> Image channels, N --> Number of frames per clip, H --> Height, W --> Width
        tensor_size = (
            (clips_per_video, 3, n_frames_to_sample, height, width)
            if self.frame_stack_format == "channel_first"
            else (clips_per_video, n_frames_to_sample, 3, height, width)
        )

        input_video = torch.zeros(
            size=tensor_size, dtype=torch.float32, device=torch.device("cpu")
        )
        return input_video

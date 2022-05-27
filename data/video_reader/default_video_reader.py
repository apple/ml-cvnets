#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
from typing import Optional, List
import torch
import av
from torch import Tensor

from . import PyAVBaseReader, register_video_reader


@register_video_reader(name="pyav_default")
class PyAVDefaultReader(PyAVBaseReader):
    """
    Default PyAv video reader

    Args:
        opts: command line arguments
        is_training (Optional[bool]): Training or validation mode. Default: `False`
    """

    def __init__(
        self, opts, is_training: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(opts=opts, is_training=is_training, *args, **kwargs)

    def read_video(
        self, filename: str, frame_ids: Optional[List] = None, *args, **kwargs
    ) -> Optional[Tensor]:

        try:
            if frame_ids is None:
                return None

            # Check basic demo for Pyav usage: https://pyav.org/docs/stable/#basic-demo
            with av.open(filename) as container:
                stream = container.streams.video[0]

                if self.fast_decoding:
                    stream.thread_type = "AUTO"

                video_frames = []
                for frame in container.decode(video=0):
                    frame_idx = frame.index
                    if frame_idx in frame_ids:
                        # using PIL so that we can apply wide range of augmentations on Frame, such as RandAug
                        frame_torch = self.frame_to_tensor(frame)
                        # check for duplicate frame ids
                        n_duplicate_frames = max(1, frame_ids.count(frame_idx))

                        video_frames.extend(
                            [copy.deepcopy(frame_torch)] * n_duplicate_frames
                        )

                # [C, H, W] x N --> [N, C, H, W] or [C, N, H, W]
                if len(video_frames) == len(frame_ids):
                    video_frames = torch.stack(video_frames, dim=self.stack_frame_dim)
                    return video_frames
                elif 0 < len(video_frames) < len(frame_ids):
                    n_delta_frames = len(frame_ids) - len(video_frames)
                    # add black frames
                    delta_frame = [torch.zeros_like(video_frames[-1])] * n_delta_frames
                    video_frames.extend(delta_frame)

                    video_frames = torch.stack(video_frames, dim=self.stack_frame_dim)
                    return video_frames
                else:
                    return None
        except av.AVError as ave_error:
            return None

    def process_video(
        self,
        vid_filename: str,
        n_frames_per_clip: Optional[int] = -1,
        clips_per_video: Optional[int] = 1,
        video_transform_fn: Optional = None,
        is_training: Optional[bool] = False,
        *args,
        **kwargs
    ):
        sampling_method = self.random_sampling if is_training else self.uniform_sampling

        total_frames = self.num_frames(filename=vid_filename)
        total_frames_to_sample = n_frames_per_clip * clips_per_video
        if n_frames_per_clip < 1:
            n_frames_per_clip = total_frames
            total_frames_to_sample = total_frames

        frame_ids = sampling_method(
            desired_frames=n_frames_per_clip,
            total_frames=total_frames,
            n_clips=clips_per_video,
        )

        # [N, C, H, W] or [C, N, H, W]
        torch_video = self.read_video(filename=vid_filename, frame_ids=frame_ids)

        if isinstance(torch_video, torch.Tensor):

            if video_transform_fn is not None:
                # Apply transformation
                torch_video = video_transform_fn({"image": torch_video})
                torch_video = torch_video["image"]

            if torch_video.shape[self.stack_frame_dim] < total_frames_to_sample:
                # This is very unlikely but, for a safer side.
                delta = total_frames_to_sample - torch_video.shape[self.stack_frame_dim]
                clip_height, clip_width = torch_video.shape[-2:]
                if self.stack_frame_dim == 0:
                    delta_frames = torch.zeros(size=(delta, 3, clip_height, clip_width))
                else:
                    delta_frames = torch.zeros(size=(3, delta, clip_height, clip_width))
                torch_video = torch.cat(
                    [torch_video, delta_frames], dim=self.stack_frame_dim
                )
            elif torch_video.shape[self.stack_frame_dim] > total_frames_to_sample:
                # truncate
                if self.stack_frame_dim == 0:
                    torch_video = torch_video[:total_frames_to_sample, ...]
                else:
                    torch_video = torch_video[:, :total_frames_to_sample, ...]

            assert torch_video.shape[self.stack_frame_dim] % clips_per_video == 0

            return self.convert_to_clips(video=torch_video, n_clips=clips_per_video)
        else:
            return None

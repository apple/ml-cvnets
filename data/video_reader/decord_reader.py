#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import sys
from typing import Dict, Optional

from data.transforms.base_transforms import BaseTransformation

try:
    import decord
except ImportError:
    pass

import av
import torch

from data.video_reader import VIDEO_READER_REGISTRY
from data.video_reader.pyav_reader import BaseAVReader
from utils import logger


@VIDEO_READER_REGISTRY.register(name="decord")
class DecordAVReader(BaseAVReader):
    """
    Video Reader using Decord.
    """

    def __init__(self, *args, **kwargs):
        if "decord" not in sys.modules:
            logger.error(
                "Decord video reader (an optional dependency) is not installed. Please"
                " run 'pip install decord'."
            )
        super().__init__(*args, **kwargs)

    def read_video(
        self,
        av_file: str,
        stream_idx: int = 0,
        audio_sample_rate: int = -1,
        custom_frame_transforms: Optional[BaseTransformation] = None,
        video_only: bool = False,
        *args,
        **kwargs
    ) -> Dict:
        video_frames = audio_frames = None
        video_fps = audio_fps = None
        decord.bridge.set_bridge("torch")
        # We have to use av package to obtain audio fps, which is not available in
        # decord.
        with av.open(str(av_file)) as container:
            available_streams = []
            for stream in container.streams:
                if stream.type == "audio":
                    # Skip audio stream if audio not required.
                    if video_only:
                        continue
                    audio_fps = container.streams.audio[0].sample_rate
                available_streams.append(stream.type)
        for stream_type in available_streams:
            if stream_type == "video":
                with open(str(av_file), "rb") as f:
                    video_reader = decord.VideoReader(f, ctx=decord.cpu(0))
                    n_video_frames = video_reader._num_frame
                    video_frames = []
                    frame_transforms = (
                        self.frame_transforms
                        if custom_frame_transforms is None
                        else custom_frame_transforms
                    )
                    for _ in range(n_video_frames):
                        video_frame = video_reader.next()  # H, W, C
                        video_frame = video_frame.permute(2, 0, 1)  # C, H, W
                        video_frame = frame_transforms({"image": video_frame})["image"]
                        video_frames.append(video_frame)
                    video_frames = torch.stack(video_frames)
                    video_fps = video_reader.get_avg_fps()
            if stream_type == "audio":
                with open(str(av_file), "rb") as f:
                    audio_reader = decord.AudioReader(
                        f, ctx=decord.cpu(0), sample_rate=audio_sample_rate
                    )
                    audio_frames = torch.tensor(audio_reader._array).transpose(0, 1)
                    audio_fps = (
                        audio_sample_rate if audio_sample_rate > 0 else audio_fps
                    )

        return {
            "audio": audio_frames,  # expected format T x C
            "video": video_frames,  # expected format T x C x H x W
            "metadata": {
                "audio_fps": audio_fps,
                "video_fps": video_fps,
                "filename": av_file,
            },
        }

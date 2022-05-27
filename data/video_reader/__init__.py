#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse
from typing import Optional

from utils.ddp_utils import is_master
from utils import logger

from .base_video_reader import PyAVBaseReader


VIDEO_READER_REGISTRY = {}


def register_video_reader(name):
    def register_video_reader_class(cls):
        if name in VIDEO_READER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate video reader class ({})".format(name)
            )

        if not issubclass(cls, PyAVBaseReader):
            raise ValueError(
                "Video reader ({}: {}) must extend PyAVBaseReader".format(
                    name, cls.__name__
                )
            )

        VIDEO_READER_REGISTRY[name] = cls
        return cls

    return register_video_reader_class


def supported_video_reader_str(video_reader_name):
    supp_list = list(VIDEO_READER_REGISTRY.keys())
    supp_str = "Video reader ({}) is not yet supported. \n Supported video readers are:".format(
        video_reader_name
    )

    for i, vr_name in enumerate(supp_list):
        supp_str += "{} \t".format(vr_name)
    logger.error(supp_str)


def general_video_reader_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Video reader", description="Arguments related to video reader"
    )
    group.add_argument(
        "--video-reader.name",
        type=str,
        default="pyav_standard",
        help="Name of video reader",
    )
    group.add_argument(
        "--video-reader.fast-video-decoding",
        action="store_true",
        help="Multi-threaded fast video decoding using pyav",
    )
    group.add_argument(
        "--video-reader.frame-stack-format",
        type=str,
        default="sequence_first",
        choices=["sequence_first", "channel_first"],
        help="Sequence first (NCHW) or channel first (CNHW) format for stacking video frames",
    )
    return parser


def arguments_video_reader(parser: argparse.ArgumentParser):
    parser = general_video_reader_args(parser=parser)

    # add video reader specific arguments
    for k, v in VIDEO_READER_REGISTRY.items():
        parser = v.add_arguments(parser=parser)

    return parser


def get_video_reader(opts, is_training: Optional[bool] = False, *args, **kwargs):
    vr_name = getattr(opts, "video_reader.name", "pyav_standard")

    is_master_node = is_master(opts)
    video_reader = None
    if vr_name in VIDEO_READER_REGISTRY:
        video_reader = VIDEO_READER_REGISTRY[vr_name](
            opts=opts, is_training=is_training
        )
    else:
        supported_video_reader_str(video_reader_name=vr_name)

    if is_master_node:
        logger.log("Video reader details: ")
        print("{}".format(video_reader))
    return video_reader


# automatically import video readers
video_reader_dir = os.path.dirname(__file__)
for file in os.listdir(video_reader_dir):
    path = os.path.join(video_reader_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        vr_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("data.video_reader." + vr_name)

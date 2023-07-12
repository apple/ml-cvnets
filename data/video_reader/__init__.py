#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

from data.video_reader.base_av_reader import BaseAVReader
from utils import logger
from utils.ddp_utils import is_master
from utils.registry import Registry

VIDEO_READER_REGISTRY = Registry(
    "video_reader",
    base_class=BaseAVReader,
    lazy_load_dirs=["data/video_reader"],
    internal_dirs=["internal", "internal/projects/*"],
)


def arguments_video_reader(parser: argparse.ArgumentParser):
    parser = BaseAVReader.add_arguments(parser=parser)

    # add video reader specific arguments
    parser = VIDEO_READER_REGISTRY.all_arguments(parser)
    return parser


def get_video_reader(opts, *args, **kwargs) -> BaseAVReader:
    """Helper function to build the video reader from command-line arguments.

    Args:
        opts: Command-line arguments
        is_training:

    Returns:
        Image projection head module.
    """

    video_reader_name = getattr(opts, "video_reader.name")

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if video_reader_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    video_reader = VIDEO_READER_REGISTRY[video_reader_name](opts, *args, **kwargs)

    is_master_node = is_master(opts)
    if is_master_node:
        logger.log("Video reader details: ")
        print("{}".format(video_reader))
    return video_reader

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.utils import data
import argparse
import time

from utils import logger
from utils.ddp_utils import is_start_rank_node, dist_barrier


class BaseZeroShotDataset(object):
    """
    Base Dataset class for Zero shot tasks
    """

    def __init__(self, opts, *args, **kwargs):
        if getattr(opts, "dataset.multi_modal_img_text.zero_shot.trove.enable", False):
            try:
                from internal.utils.server_utils import load_from_data_server

                opts = load_from_data_server(
                    opts=opts, is_training=False, arg_prefix="dataset.multi_modal_img_text.zero_shot"
                )
            except Exception as e:
                logger.error("Unable to load from the server. Error: {}".format(str(e)))

        root = getattr(opts, "dataset.multi_modal_img_text.zero_shot.root_val", None)
        self.root = root
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def class_names():
        pass

    def __repr__(self):
        return "{}(root={})".format(self.__class__.__name__, self.root)

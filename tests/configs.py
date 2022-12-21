#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys

sys.path.append("..")

from options.utils import load_config_file
from options.opts import get_training_arguments


def get_config(config_file: str = None):
    parser = get_training_arguments(parse_args=False)
    opts = parser.parse_args([])
    setattr(opts, "common.config_file", config_file)
    opts = load_config_file(opts)
    return opts

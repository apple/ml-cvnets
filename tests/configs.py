#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

from options.opts import get_training_arguments
from options.utils import load_config_file


def get_config(
    config_file: str = None, disable_ddp_distributed: bool = True
) -> argparse.Namespace:
    """Produces a resolved config (i.e. opts) object to be used in tests.

    Args:
        config_file: If provided, the contents of the @config_file path will override
          the default configs.
        disable_ddp_distributed: ``ddp.distributed`` config entry is not defined in
          the parser, but rather set by the entrypoints on the fly based on the
          availability of multiple gpus. In the tests, we usually don't want to use
          ``ddp.distributed``, even if multiple gpus are available.
    """
    parser = get_training_arguments(parse_args=False)
    opts = parser.parse_args([])
    setattr(opts, "common.config_file", config_file)
    opts = load_config_file(opts)

    if disable_ddp_distributed:
        setattr(opts, "ddp.use_distributed", False)

    return opts


# If slow, this can be turned into a "session"-scoped fixture
# @pytest.fixture(scope='session')
def default_training_opts() -> argparse.Namespace:
    opts = get_training_arguments(args=[])
    return opts

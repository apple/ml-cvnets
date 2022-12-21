#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import collections
import os

import yaml

from utils import logger
from utils.ddp_utils import is_master
from utils.download_utils import get_local_path

try:
    # Workaround for DeprecationWarning when importing Collections
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

DEFAULT_CONFIG_DIR = "config"


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(opts):
    config_file_name = getattr(opts, "common.config_file", None)
    if config_file_name is None:
        return opts
    is_master_node = is_master(opts)

    if is_master_node:
        config_file_name = get_local_path(opts=opts, path=config_file_name)

    if not os.path.isfile(config_file_name):
        if len(config_file_name.split("/")) == 1:
            # loading files from default config folder
            new_config_file_name = "{}/{}".format(DEFAULT_CONFIG_DIR, config_file_name)
            if not os.path.isfile(new_config_file_name) and is_master_node:
                logger.error(
                    "Configuration file neither exists at {} nor at {}".format(
                        config_file_name, new_config_file_name
                    )
                )
            else:
                config_file_name = new_config_file_name
        else:
            # If absolute path of the file is passed
            if not os.path.isfile(config_file_name) and is_master_node:
                logger.error(
                    "Configuration file does not exists at {}".format(config_file_name)
                )

    setattr(opts, "common.config_file", config_file_name)
    with open(config_file_name, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
        except yaml.YAMLError as exc:
            if is_master_node:
                logger.error(
                    "Error while loading config file: {}. Error message: {}".format(
                        config_file_name, str(exc)
                    )
                )

    # override arguments
    override_args = getattr(opts, "override_args", None)
    if override_args is not None:
        for override_k, override_v in override_args.items():
            if hasattr(opts, override_k):
                setattr(opts, override_k, override_v)

    return opts


def extend_selected_args_with_prefix(
    parser: argparse.ArgumentParser, check_string: str, add_prefix: str
) -> argparse.ArgumentParser:
    """
    Helper function to add a prefix to certain arguments.
    An example use case is distillation, where we want to add --teacher as a prefix to all --model.* arguments
    """
    # all arguments are stored as actions
    options = parser._actions

    for option in options:
        option_strings = option.option_strings
        # option strings are stored as a list
        for option_string in option_strings:
            if option_string.split(".")[0] == check_string:
                parser.add_argument(
                    add_prefix + option.dest.replace("_", "-"),
                    nargs="?"
                    if isinstance(option, argparse._StoreTrueAction)
                    else option.nargs,
                    const=option.const,
                    default=option.default,
                    type=option.type,
                    choices=option.choices,
                    help=option.help,
                    metavar=option.metavar,
                )
    return parser

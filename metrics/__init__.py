#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import importlib
import argparse

SUPPORTED_STATS = ['loss']


def register_stats_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_STATS:
            raise ValueError("Cannot register duplicate state ({})".format(name))
        SUPPORTED_STATS.append(name)
        return fn
    return register_fn


def arguments_stats(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Statistics", description="Statistics")
    group.add_argument('--stats.name', type=str, default=['loss'], nargs="+", help="Name of statistics")
    group.add_argument('--stats.checkpoint-metric', type=str, default=None, help="Metric to use for saving checkpoints")
    group.add_argument('--stats.checkpoint-metric-max', type=str, default=None,
                       help="Maximize checkpoint metric")

    return parser


# automatically import different metrics
metrics_dir = os.path.dirname(__file__)
for file in os.listdir(metrics_dir):
    path = os.path.join(metrics_dir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("metrics." + model_name)


from metrics.stats import Statistics
from metrics.metric_monitor import metric_monitor
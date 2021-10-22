#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import sys
import time
import numpy as np
import torch
from utils import logger
from typing import Optional, Dict, Union, Any

from . import SUPPORTED_STATS


class Statistics(object):
    def __init__(self, metric_names: Optional[list] = ['loss'], is_master_node: Optional[bool] = False) -> None:
        if len(metric_names) == 0:
            logger.error('Metric names list cannot be empty')

        # key is the metric name and value is the value
        metric_dict: Dict[str, Union[Any]] = {}
        metric_counters = {}
        for m_name in metric_names:
            if m_name in SUPPORTED_STATS:
                metric_dict[m_name] = None
                metric_counters[m_name] = 0
            else:
                if is_master_node:
                    logger.log('{} statistics not supported. Supported: {}'.format(m_name, SUPPORTED_STATS))

        self.metric_dict = metric_dict
        self.supported_metrics = list(metric_dict.keys())
        self.metric_counters = metric_counters
        self.round_places = 4
        self.is_master_node = is_master_node

        self.batch_time = 0
        self.batch_counter = 0

    def update(self, metric_vals: dict, batch_time: float, n: Optional[int] = 1) -> None:
        for k, v in metric_vals.items():
            if k in self.supported_metrics:
                if self.metric_dict[k] is None:
                    if k == "iou":
                        self.metric_dict[k] = {"inter": v["inter"] * n, "union": v["union"] * n}
                    else:
                        self.metric_dict[k] = v * n
                else:
                    if k == "iou":
                        self.metric_dict[k]["inter"] += (v["inter"] * n)
                        self.metric_dict[k]["union"] += (v["union"] * n)
                    else:
                        self.metric_dict[k] += v * n
                self.metric_counters[k] += n
        self.batch_time += batch_time
        self.batch_counter += 1

    def avg_statistics_all(self, sep=": ") -> list:
        metric_stats = []
        for k, v in self.metric_dict.items():
            counter = self.metric_counters[k]

            if k == "iou":
                inter = (v["inter"] * 1.0) / counter
                union = (v["union"] * 1.0) / counter
                iou = inter / union
                if isinstance(iou, torch.Tensor):
                    iou = iou.cpu().numpy()
                # Converting iou from [0, 1] to [0, 100]
                # other metrics are by default in [0, 100 range]
                v_avg = np.mean(iou) * 100.0
            else:
                v_avg = (v * 1.0) / counter

            v_avg = round(v_avg, self.round_places)

            metric_stats.append("{:<}{}{:.4f}".format(k, sep, v_avg))
        return metric_stats

    def avg_statistics(self, metric_name: str) -> float:
        avg_val = None
        if metric_name in self.supported_metrics:
            counter = self.metric_counters[metric_name]
            v = self.metric_dict[metric_name]

            if metric_name == "iou":
                inter = (v["inter"] * 1.0) / counter
                union = (v["union"] * 1.0) / counter
                iou = inter / union
                if isinstance(iou, torch.Tensor):
                    iou = iou.cpu().numpy()
                # Converting iou from [0, 1] to [0, 100]
                # other metrics are by default in [0, 100 range]
                avg_val = np.mean(iou) * 100.0
            else:
                avg_val = (v * 1.0) / counter

            avg_val = round(avg_val, self.round_places)
        return avg_val

    def iter_summary(self,
                     epoch: int,
                     n_processed_samples: int,
                     total_samples: int,
                     elapsed_time: float,
                     learning_rate: float or list) -> None:
        if self.is_master_node:
            metric_stats = self.avg_statistics_all()
            el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)
            if isinstance(learning_rate, float):
                lr_str = "LR: {:1.6f}".format(learning_rate)
            else:
                learning_rate = [round(lr, 6) for lr in learning_rate]
                lr_str = "LR: {}".format(learning_rate)
            epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(epoch, n_processed_samples, total_samples)
            batch_str = "Avg. batch load time: {:1.3f}".format(self.batch_time / self.batch_counter)

            stats_summary = [epoch_str]
            stats_summary.extend(metric_stats)
            stats_summary.append(lr_str)
            stats_summary.append(batch_str)
            stats_summary.append(el_time_str)

            summary_str = ", ".join(stats_summary)
            logger.log(summary_str)
            sys.stdout.flush()

    def epoch_summary(self, epoch: int, stage: Optional[str] = "Training") -> None:
        if self.is_master_node:
            metric_stats = self.avg_statistics_all(sep="=")
            metric_stats_str = " || ".join(metric_stats)
            logger.log('*** {} summary for epoch {}'.format(stage.title(), epoch))
            print("\t {}".format(metric_stats_str))
            sys.stdout.flush()

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import sys
import time
import numpy as np
import torch
from utils import logger
from typing import Optional, Dict, Union, Any, List
from numbers import Number

from . import SUPPORTED_STATS


class Statistics(object):
    def __init__(
        self,
        metric_names: Optional[list] = ["loss"],
        is_master_node: Optional[bool] = False,
    ) -> None:
        if len(metric_names) == 0:
            logger.error("Metric names list cannot be empty")

        # key is the metric name and value is the value
        metric_dict: Dict[str, Union[Any]] = {}
        metric_counters = {}
        for m_name in metric_names:
            # Don't use coco_map key here as it is handled separately
            if m_name == "coco_map":
                continue

            if m_name in SUPPORTED_STATS:
                metric_dict[m_name] = None
                metric_counters[m_name] = 0
            else:
                if is_master_node:
                    logger.log(
                        "{} statistics not supported. Supported: {}".format(
                            m_name, SUPPORTED_STATS
                        )
                    )

        self.metric_dict = metric_dict
        self.supported_metrics = list(metric_dict.keys())
        self.metric_counters = metric_counters
        self.round_places = 4
        self.is_master_node = is_master_node

        self.batch_time = 0
        self.batch_counter = 0

    def update(
        self, metric_vals: dict, batch_time: float, n: Optional[int] = 1
    ) -> None:
        for k, v in metric_vals.items():
            if k in self.supported_metrics:
                if self.metric_dict[k] is None:
                    if k == "iou":
                        if isinstance(v["inter"], np.ndarray):
                            self.metric_dict[k] = {
                                "inter": v["inter"] * n,
                                "union": v["union"] * n,
                            }
                        else:
                            logger.error(
                                "IOU computation is only supported using np.ndarray."
                            )
                    elif isinstance(v, Dict):
                        self.metric_dict[k] = dict()
                        for k1, v1 in v.items():
                            self.metric_dict[k][k1] = v1 * n
                    elif isinstance(v, Number):
                        self.metric_dict[k] = v * n
                    else:
                        logger.error(
                            "Dict[str, float] or float are supported in {}".format(
                                self.__class__.__name__
                            )
                        )
                else:
                    if k == "iou":
                        if isinstance(v["inter"], np.ndarray):
                            self.metric_dict[k]["inter"] += v["inter"] * n
                            self.metric_dict[k]["union"] += v["union"] * n
                        else:
                            logger.error(
                                "IOU computation is only supported using np.ndarray."
                            )
                    elif isinstance(v, Dict):
                        for k1, v1 in v.items():
                            self.metric_dict[k][k1] += v1 * n
                    elif isinstance(v, Number):
                        self.metric_dict[k] += v * n
                    else:
                        logger.error(
                            "Dict[str, float] or Number are supported in {}".format(
                                self.__class__.__name__
                            )
                        )

                self.metric_counters[k] += n
        self.batch_time += batch_time
        self.batch_counter += 1

    def avg_statistics_all(self, sep=": ") -> List[str]:
        """
        This function computes average statistics of all metrics and returns them as a list of strings.

        Examples:
         loss: 12.9152
         loss: {'total_loss': 12.9152, 'reg_loss': 2.8199, 'cls_loss': 10.0953}
        """

        metric_stats = []
        for k, v in self.metric_dict.items():
            counter = self.metric_counters[k]

            if k == "iou":
                if isinstance(v["inter"], np.ndarray):
                    inter = (v["inter"] * 1.0) / counter
                    union = (v["union"] * 1.0) / counter
                    iou = inter / union
                    if isinstance(iou, torch.Tensor):
                        iou = iou.cpu().numpy()
                    # Converting iou from [0, 1] to [0, 100]
                    # other metrics are by default in [0, 100 range]
                    v_avg = np.mean(iou) * 100.0
                    v_avg = round(v_avg, self.round_places)
                else:
                    logger.error("IOU computation is only supported using np.ndarray.")
            elif isinstance(v, Dict):
                v_avg = {}
                for k1, v1 in v.items():
                    v_avg[k1] = round((v1 * 1.0) / counter, self.round_places)
            else:
                v_avg = round((v * 1.0) / counter, self.round_places)

            metric_stats.append("{:<}{}{}".format(k, sep, v_avg))
        return metric_stats

    def avg_statistics(
        self, metric_name: str, sub_metric_name: Optional[str] = None, *args, **kwargs
    ) -> float:
        """
        This function computes the average statistics of a given metric.

        .. note::
        The statistics are stored in form of a dictionary and each key-value pair can be of string and number
        OR string and dictionary of string and number.

        Examples:
             {'loss': 10.0, 'top-1': 50.0}
             {'loss': {'total_loss': 10.0, 'cls_loss': 2.0, 'reg_loss': 8.0}, 'mAP': 5.0}

        """
        avg_val = None
        if metric_name in self.supported_metrics:
            counter = self.metric_counters[metric_name]
            v = self.metric_dict[metric_name]

            if metric_name == "iou":
                if isinstance(v["inter"], np.ndarray):
                    inter = (v["inter"] * 1.0) / counter
                    union = (v["union"] * 1.0) / counter
                    iou = inter / union
                    if isinstance(iou, torch.Tensor):
                        iou = iou.cpu().numpy()
                    # Converting iou from [0, 1] to [0, 100]
                    # other metrics are by default in [0, 100 range]
                    avg_val = np.mean(iou) * 100.0
                    avg_val = round(avg_val, self.round_places)
                else:
                    logger.error("IOU computation is only supported using np.ndarray.")

            elif isinstance(v, Dict) and sub_metric_name is not None:
                sub_metric_keys = list(v.keys())
                if sub_metric_name in sub_metric_keys:
                    avg_val = round(
                        (v[sub_metric_name] * 1.0) / counter, self.round_places
                    )
                else:
                    logger.error(
                        "{} not present in the dictionary. Available keys are: {}".format(
                            sub_metric_name, sub_metric_keys
                        )
                    )
            elif isinstance(v, Number):
                avg_val = round((v * 1.0) / counter, self.round_places)

        return avg_val

    def iter_summary(
        self,
        epoch: int,
        n_processed_samples: int,
        total_samples: int,
        elapsed_time: float,
        learning_rate: float or list,
    ) -> None:
        if self.is_master_node:
            metric_stats = self.avg_statistics_all()
            el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)
            if isinstance(learning_rate, float):
                lr_str = "LR: {:1.6f}".format(learning_rate)
            else:
                learning_rate = [round(lr, 6) for lr in learning_rate]
                lr_str = "LR: {}".format(learning_rate)
            epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(
                epoch, n_processed_samples, total_samples
            )
            batch_str = "Avg. batch load time: {:1.3f}".format(
                self.batch_time / self.batch_counter
            )

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
            logger.log("*** {} summary for epoch {}".format(stage.title(), epoch))
            print("\t {}".format(metric_stats_str))
            sys.stdout.flush()

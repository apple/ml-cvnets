#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import argparse
from typing import Any


class BaseCriteria(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseCriteria, self).__init__()
        self.eps = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(self, input_sample: Tensor, prediction: Any, target: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _class_weights(target: Tensor, n_classes: int, norm_val: float = 1.1) -> Tensor:
        class_hist: Tensor = torch.histc(target.float(), bins=n_classes, min=0, max=n_classes - 1)
        mask_indices = (class_hist == 0)

        # normalize between 0 and 1 by dividing by the sum
        norm_hist = torch.div(class_hist, class_hist.sum())
        norm_hist = torch.add(norm_hist, norm_val)

        # compute class weights..
        # samples with more frequency will have less weight and vice-versa
        class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))

        # mask the classes which do not have samples in the current batch
        class_wts[mask_indices] = 0.0

        return class_wts.to(device=target.device)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
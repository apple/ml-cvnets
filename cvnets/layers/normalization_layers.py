#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
from typing import Optional
from utils import logger
import math
from .identity import Identity
from .normalization import (
    BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm, SUPPORTED_NORM_FNS
)


norm_layers_tuple = (BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm)


def get_normalization_layer(opts, num_features: int, norm_type: Optional[str] = None, num_groups: Optional[int] = None,
                            **kwargs):
    norm_type = getattr(opts, "model.normalization.name", "batch_norm") if norm_type is None else norm_type
    num_groups = getattr(opts, "model.normalization.groups", 1) if num_groups is None else num_groups
    momentum = getattr(opts, "model.normalization.momentum", 0.1)

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type in ['batch_norm', 'batch_norm_2d']:
        norm_layer = BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'batch_norm_1d':
        norm_layer = BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['sync_batch_norm', 'sbn']:
        norm_layer = SyncBatchNorm(num_features=num_features, momentum=momentum)
    elif norm_type in ['group_norm', 'gn']:
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type in ['instance_norm', 'instance_norm_2d']:
        norm_layer = InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == "instance_norm_1d":
        norm_layer = InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['layer_norm', 'ln']:
        norm_layer = LayerNorm(num_features)
    elif norm_type == 'identity':
        norm_layer = Identity()
    else:
        logger.error(
            'Supported normalization layer arguments are: {}. Got: {}'.format(SUPPORTED_NORM_FNS, norm_type))
    return norm_layer


class AdjustBatchNormMomentum(object):
    # TODO: Experimental
    round_places = 6

    def __init__(self, opts, ):
        self.is_iteration_based = getattr(opts, "scheduler.is_iteration_based", True)
        self.warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 10000)

        if self.is_iteration_based:
            self.max_steps = getattr(opts, "scheduler.max_iterations", 100000)
            self.max_steps -= self.warmup_iterations
            assert self.max_steps > 0
        else:
            logger.warning("Running {} for epoch-based methods. Not yet validation.".format(self.__class__.__name__))
            self.max_steps = getattr(opts, "scheduler.max_epochs", 100)

        self.momentum = getattr(opts, "model.normalization.momentum", 0.1)
        self.min_momentum = getattr(opts, "adjust_bn_momentum.final_momentum_value", 1e-6)
        assert self.min_momentum < self.momentum
        anneal_method = getattr(opts, "adjust_bn_momentum.anneal_type", "cosine").lower()

        if anneal_method == "cosine":
            self.anneal_fn = self._cosine
        elif anneal_method == "linear":
            self.anneal_fn = self._linear
        else:
            raise RuntimeError("Anneal method ({}) not yet implemented".format(anneal_method))
        self.anneal_method = anneal_method

    def _cosine(self, step: int) -> float:
        curr_momentum = self.min_momentum + 0.5 * (self.momentum - self.min_momentum) * \
                  (1 + math.cos(math.pi * step / self.max_steps))

        return round(curr_momentum, self.round_places)

    def _linear(self, step: int) -> float:
        momentum_step = (self.momentum - self.min_momentum) / self.max_steps
        curr_momentum = self.momentum - (step * momentum_step)
        return round(curr_momentum, self.round_places)

    def adjust_momentum(self, model: nn.Module, iteration: int, epoch: int) -> None:
        if iteration >= self.warmup_iterations:
            step = iteration - self.warmup_iterations if self.is_iteration_based else epoch
            curr_momentum = max(0.0, self.anneal_fn(step))

            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.training:
                    m.momentum = curr_momentum

    def __repr__(self):
        return "{}(\n\t " \
               "iteration_based={} " \
               "\n\t inital_momentum={} " \
               "\n\t final_momentum={} " \
               "\n\t anneal_method={} " \
               "\n)".format(self.__class__.__name__,
                            self.is_iteration_based,
                            self.momentum,
                            self.min_momentum,
                            self.anneal_method)
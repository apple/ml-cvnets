#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import math

from torch import nn

from cvnets.layers.normalization import NORM_LAYER_CLS, build_normalization_layer
from utils import logger

norm_layers_tuple = tuple(NORM_LAYER_CLS)


get_normalization_layer = build_normalization_layer


class AdjustBatchNormMomentum(object):
    """
    This class enables adjusting the momentum in batch normalization layer.

    .. note::
        It's an experimental feature and should be used with caution.
    """

    round_places = 6

    def __init__(self, opts, *args, **kwargs):
        self.is_iteration_based = getattr(opts, "scheduler.is_iteration_based", True)
        self.warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 10000)

        if self.is_iteration_based:
            self.max_steps = getattr(opts, "scheduler.max_iterations", 100000)
            self.max_steps -= self.warmup_iterations
            assert self.max_steps > 0
        else:
            logger.warning(
                "Running {} for epoch-based methods. Not yet validation.".format(
                    self.__class__.__name__
                )
            )
            self.max_steps = getattr(opts, "scheduler.max_epochs", 100)

        self.momentum = getattr(opts, "model.normalization.momentum", 0.1)
        self.min_momentum = getattr(
            opts, "model.normalization.adjust_bn_momentum.final_momentum_value", 1e-6
        )

        if self.min_momentum >= self.momentum:
            logger.error(
                "Min. momentum value in {} should be <= momentum. Got {} and {}".format(
                    self.__class__.__name__, self.min_momentum, self.momentum
                )
            )

        anneal_method = getattr(
            opts, "model.normalization.adjust_bn_momentum.anneal_type", "cosine"
        )
        if anneal_method is None:
            logger.warning(
                "Annealing method in {} is None. Setting to cosine".format(
                    self.__class__.__name__
                )
            )
            anneal_method = "cosine"

        anneal_method = anneal_method.lower()

        if anneal_method == "cosine":
            self.anneal_fn = self._cosine
        elif anneal_method == "linear":
            self.anneal_fn = self._linear
        else:
            raise RuntimeError(
                "Anneal method ({}) not yet implemented".format(anneal_method)
            )
        self.anneal_method = anneal_method

    def _cosine(self, step: int) -> float:
        curr_momentum = self.min_momentum + 0.5 * (
            self.momentum - self.min_momentum
        ) * (1 + math.cos(math.pi * step / self.max_steps))

        return round(curr_momentum, self.round_places)

    def _linear(self, step: int) -> float:
        momentum_step = (self.momentum - self.min_momentum) / self.max_steps
        curr_momentum = self.momentum - (step * momentum_step)
        return round(curr_momentum, self.round_places)

    def adjust_momentum(self, model: nn.Module, iteration: int, epoch: int) -> None:
        if iteration >= self.warmup_iterations:
            step = (
                iteration - self.warmup_iterations if self.is_iteration_based else epoch
            )
            curr_momentum = max(0.0, self.anneal_fn(step))

            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and m.training:
                    m.momentum = curr_momentum

    def __repr__(self):
        return "{}(iteration_based={}, inital_momentum={}, final_momentum={}, anneal_method={})".format(
            self.__class__.__name__,
            self.is_iteration_based,
            self.momentum,
            self.min_momentum,
            self.anneal_method,
        )

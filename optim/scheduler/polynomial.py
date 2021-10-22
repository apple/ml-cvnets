#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse

from . import register_scheduler
from .base_scheduler import BaseLRScheduler


@register_scheduler("polynomial")
class PolynomialScheduler(BaseLRScheduler):
    """
        Polynomial LR scheduler
    """
    def __init__(self, opts, **kwargs) -> None:
        is_iter_based = getattr(opts, "scheduler.is_iteration_based", False)
        max_iterations = getattr(opts, "scheduler.max_iterations", 50000)
        warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 0)
        max_epochs = getattr(opts, "scheduler.max_epochs", 350)

        super(PolynomialScheduler, self).__init__(opts=opts)

        self.start_lr = getattr(opts, "scheduler.polynomial.start_lr", 0.1)
        self.end_lr = getattr(opts, "scheduler.polynomial.end_lr", 1e-5)
        self.power = getattr(opts, "scheduler.polynomial.power", 2.5)

        self.warmup_iterations = max(warmup_iterations, 0)
        if self.warmup_iterations > 0:
            warmup_init_lr = getattr(opts, "scheduler.warmup_init_lr", 1e-7)
            self.warmup_init_lr = warmup_init_lr
            self.warmup_step = (self.start_lr - self.warmup_init_lr) / self.warmup_iterations

        self.is_iter_based = is_iter_based
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title="Polynomial LR arguments", description="Polynomial LR arguments")

        group.add_argument('--scheduler.polynomial.power', type=float, default=2.5, help="Polynomial power")
        group.add_argument('--scheduler.polynomial.start-lr', type=float, default=0.1, help="Start LR in Poly LR scheduler")
        group.add_argument('--scheduler.polynomial.end-lr', type=float, default=1e-5, help="End LR in Poly LR scheduler")

        return parser

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            curr_iter = curr_iter - self.warmup_iterations
            factor = (curr_iter / self.max_iterations) if self.is_iter_based else (epoch / self.max_epochs)
            curr_lr = (self.start_lr - self.end_lr) * ((1.0 - factor) ** self.power) + self.end_lr
        return max(0.0, curr_lr)

    def __repr__(self) -> str:
        repr_str = '{}('.format(self.__class__.__name__)
        repr_str += '\n\tpower={}\n\tstart_lr={}'.format(self.power, self.start_lr)
        if self.end_lr > 0:
            repr_str += '\n\tend_lr={}'.format(self.end_lr)

        if self.warmup_iterations > 0:
            repr_str += '\n\twarmup_init_lr={}\n\twarmup_iters={}'.format(self.warmup_init_lr, self.warmup_iterations)

        repr_str += '\n )'
        return repr_str
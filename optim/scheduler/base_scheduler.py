#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse


class BaseLRScheduler(object):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts
        self.round_places = 8
        self.lr_multipliers = getattr(opts, "optim.lr_multipliers", None)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def get_lr(self, epoch: int, curr_iter: int):
        raise NotImplementedError

    def update_lr(self, optimizer, epoch: int, curr_iter: int):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = round(lr * self.lr_multipliers[g_id], self.round_places)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(lr, self.round_places)
        return optimizer

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list
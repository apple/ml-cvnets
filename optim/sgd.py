#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
from torch.optim import SGD

from . import register_optimizer
from .base_optim import BaseOptim


@register_optimizer("sgd")
class SGDOptimizer(BaseOptim, SGD):
    """
        SGD: http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    """
    def __init__(self, opts, model_params) -> None:
        BaseOptim.__init__(self, opts=opts)
        nesterov = getattr(opts, "optim.sgd.nesterov", False)
        momentum = getattr(opts, "optim.sgd.momentum", 0.9)

        SGD.__init__(
            self,
            params=model_params,
            lr=self.lr,
            momentum=momentum,
            weight_decay=self.weight_decay,
            nesterov=nesterov
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group('SGD arguments', 'SGD arguments')
        group.add_argument('--optim.sgd.momentum', default=0.9, type=float, help='Momemtum in SGD')
        group.add_argument('--optim.sgd.nesterov', action='store_true', help='Use nesterov in SGD')
        return parser

    def __repr__(self) -> str:
        group_dict = dict()
        for i, group in enumerate(self.param_groups):
            for key in sorted(group.keys()):
                if key == 'params':
                    continue
                if key not in group_dict:
                    group_dict[key] = [group[key]]
                else:
                    group_dict[key].append(group[key])

        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        for k, v in group_dict.items():
            format_string += '\t {0}: {1}\n'.format(k, v)
        format_string += ')'
        return format_string
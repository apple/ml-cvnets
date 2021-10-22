#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
from torch.optim import AdamW

from . import register_optimizer
from .base_optim import BaseOptim


@register_optimizer("adamw")
class AdamWOptimizer(BaseOptim, AdamW):
    """
        AdamW: https://arxiv.org/abs/1711.05101
    """
    def __init__(self, opts, model_params) -> None:
        BaseOptim.__init__(self, opts=opts)
        beta1 = getattr(opts, "optim.adamw.beta1", 0.9)
        beta2 = getattr(opts, "optim.adamw.beta2", 0.98)
        ams_grad = getattr(opts, "optim.adamw.amsgrad", False)
        AdamW.__init__(self, params=model_params, lr=self.lr, betas=(beta1, beta2), eps=self.eps,
                       weight_decay=self.weight_decay, amsgrad=ams_grad)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group('AdamW arguments', 'AdamW arguments')
        group.add_argument('--optim.adamw.beta1', type=float, default=0.9, help='Adam Beta1')
        group.add_argument('--optim.adamw.beta2', type=float, default=0.98, help='Adam Beta2')
        group.add_argument('--optim.adamw.amsgrad', action='store_true', help='Use AMSGrad in ADAM')
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
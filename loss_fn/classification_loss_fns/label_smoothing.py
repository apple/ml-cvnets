#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import nn, Tensor
import argparse

from . import register_classification_loss_fn
from .. import BaseCriteria


@register_classification_loss_fn(name="label_smoothing")
class LabelSmoothing(BaseCriteria):
    '''
        Adapted from Fairseq:
            https://github.com/pytorch/fairseq/blob/main/fairseq/criterions/label_smoothed_cross_entropy.py
    '''

    def __init__(self, opts, reduce=True, reduction='mean', *args, **kwargs):
        smoothing = getattr(opts, "loss.classification.label_smoothing_factor", 0.1)
        ignore_idx = getattr(opts, "loss.ignore_idx", -1)

        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_idx = ignore_idx
        self.reduce = reduce
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=-1)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.classification.label-smoothing-factor", type=float, default=0.1,
                           help="Label smoothing value")
        return parser

    def _compute_loss(self, log_probs, target):
        if target.dim() == log_probs.dim() - 1:
            target = target.unsqueeze(-1)

        pad_mask = target.eq(self.ignore_idx) if self.ignore_idx is not None else None
        if pad_mask is not None and pad_mask.any():
            target[pad_mask] = 0
            nll_loss = -log_probs.gather(dim=-1, index=target)
            smooth_loss = -log_probs.sum(dim=-1, keepdim=True)

            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = -log_probs.gather(dim=-1, index=target)
            smooth_loss = -log_probs.sum(dim=-1, keepdim=True)

            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.smoothing / log_probs.size(-1)
        loss = self.confidence * nll_loss + eps_i * smooth_loss
        return loss

    def forward(self, input_sample: Tensor, prediction: Tensor, target: Tensor) -> Tensor:
        if self.training:
            assert prediction.dim() == 2, 'Should be B x C'
            batch_size, num_classes = prediction.size()
            log_probs = F.log_softmax(prediction, dim=-1)
            log_probs = log_probs.view(-1, num_classes)
            target = target.view(-1, 1)
            loss = self._compute_loss(log_probs, target)
            if self.reduction == 'mean':
                loss /= batch_size
            return loss
        else:
            return F.cross_entropy(input=prediction, target=target)

    def __repr__(self):
        return "{}(\n\t ignore_idx={} \n\t label_smooth={}\n)".format(
            self.__class__.__name__,
            self.ignore_idx,
            self.smoothing
        )

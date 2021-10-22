#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Union
import argparse

from . import register_segmentation_loss_fn
from .. import BaseCriteria


@register_segmentation_loss_fn(name="cross_entropy")
class SegCrossEntropy(BaseCriteria):
    def __init__(self, opts):
        super(SegCrossEntropy, self).__init__()
        ignore_idx = getattr(opts, "loss.ignore_idx", -1)
        use_cls_wts = getattr(opts, "loss.segmentation.cross_entropy_class_weights", False)
        self.ignore_idx = ignore_idx
        self.weighted_loss = use_cls_wts
        self.aux_wt = getattr(opts, "loss.segmentation.cross_entropy_aux_weight", 0.4)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.segmentation.cross-entropy-class-weights", action="store_true",
                           help="Use class weights in loss function")
        group.add_argument("--loss.segmentation.cross-entropy-aux-weight", type=float, default=0.4,
                           help="Weight of auxiliary loss")
        return parser

    def _compute_loss(self, pred_mask, target_mask, weight=None):
        b, c, x_h, x_w = pred_mask.shape
        b, y_h, y_w = target_mask.shape
        if x_h != y_h or x_w != y_w:
            pred_mask = F.interpolate(pred_mask, size=(y_h, y_w), mode='nearest')
            return F.cross_entropy(input=pred_mask, target=target_mask, weight=weight, ignore_index=self.ignore_idx)
        else:
            return F.cross_entropy(input=pred_mask, target=target_mask, weight=weight, ignore_index=self.ignore_idx)

    def forward(self, input_sample: Tensor, prediction: Union[Tensor or Tuple[Tensor, Tensor]], target: Tensor) -> Tensor:
        aux_out = None
        if isinstance(prediction, Tuple) and len(prediction) == 2:
            mask, aux_out = prediction
            assert isinstance(mask, Tensor)
            assert isinstance(aux_out, Tensor)
        elif isinstance(prediction, Tensor):
            mask = prediction
            assert isinstance(mask, Tensor)
        else:
            raise NotImplementedError("For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor")

        cls_wts = None
        if self.training:
            if self.weighted_loss:
                n_classes = mask.size(1)  # Mask is of shape B x C x H x W
                cls_wts = self._class_weights(target=target, n_classes=n_classes)
            total_loss = self._compute_loss(pred_mask=mask, target_mask=target, weight=cls_wts)

            if aux_out is not None:
                total_loss += (self.aux_wt * self._compute_loss(pred_mask=aux_out, target_mask=target, weight=cls_wts))
                total_loss *= 0.5
            return total_loss
        else:
            return self._compute_loss(pred_mask=mask, target_mask=target, weight=cls_wts)

    def __repr__(self):
        repr_str = "{}(\n\tweighted_loss={} \n\tignore_idx={}".format(
            self.__class__.__name__,
            self.weighted_loss,
            self.ignore_idx
        )

        if self.aux_wt > 0:
            repr_str += "\n\taux_wt={}".format(self.aux_wt)
        return repr_str + "\n)"

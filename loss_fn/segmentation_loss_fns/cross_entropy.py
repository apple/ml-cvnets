#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Union
import argparse

from . import register_segmentation_loss_fn
from .. import BaseCriteria


@register_segmentation_loss_fn(name="cross_entropy")
class SegCrossEntropy(BaseCriteria):
    """Cross entropy loss for the task of semantic segmentation"""

    def __init__(self, opts):
        super(SegCrossEntropy, self).__init__()
        self.ignore_idx = getattr(opts, "loss.ignore_idx", -1)
        self.weighted_loss = getattr(
            opts, "loss.segmentation.cross_entropy.class_weights", False
        )
        self.aux_wt = getattr(opts, "loss.segmentation.cross_entropy.aux_weight", 0.4)
        self.label_smoothing = getattr(
            opts, "loss.segmentation.cross_entropy.label_smoothing", 0.0
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.class-weights",
            action="store_true",
            help="Use class weights in loss function",
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.aux-weight",
            type=float,
            default=0.4,
            help="Weight of auxiliary loss",
        )
        group.add_argument(
            "--loss.segmentation.cross-entropy.label-smoothing",
            type=float,
            default=0.0,
            help="Label smoothing in CE loss for the task of segmentation",
        )

        return parser

    def _compute_loss(self, pred_mask, target_mask, weight=None):
        b, c, x_h, x_w = pred_mask.shape
        b, y_h, y_w = target_mask.shape

        # use label smoothing only for training
        label_smoothing = self.label_smoothing if self.training else 0.0

        if x_h != y_h or x_w != y_w:
            pred_mask = F.interpolate(
                pred_mask, size=(y_h, y_w), mode="bilinear", align_corners=True
            )

        loss = F.cross_entropy(
            input=pred_mask,
            target=target_mask,
            weight=weight,
            ignore_index=self.ignore_idx,
            label_smoothing=label_smoothing,
        )

        return loss

    def forward(
        self,
        input_sample: Tensor,
        prediction: Union[Tensor or Tuple[Tensor, Tensor]],
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        aux_out = None
        if isinstance(prediction, Tuple) and len(prediction) == 2:
            mask, aux_out = prediction
            assert isinstance(mask, Tensor)
            assert isinstance(aux_out, Tensor)
        elif isinstance(prediction, Tensor):
            mask = prediction
            assert isinstance(mask, Tensor)
        else:
            raise NotImplementedError(
                "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor"
            )

        cls_wts = None
        if self.training:
            if self.weighted_loss:
                n_classes = mask.size(1)  # Mask is of shape B x C x H x W
                cls_wts = self._class_weights(target=target, n_classes=n_classes)
            total_loss = self._compute_loss(
                pred_mask=mask, target_mask=target, weight=cls_wts
            )

            if aux_out is not None:
                loss_aux = self._compute_loss(
                    pred_mask=aux_out, target_mask=target, weight=cls_wts
                )
                total_loss = total_loss + (self.aux_wt * loss_aux)
            return total_loss
        else:
            return self._compute_loss(pred_mask=mask, target_mask=target, weight=None)

    def __repr__(self):
        repr_str = (
            "{}(\n\tweighted_loss={}\n\tignore_idx={}\n\tlabel_smoothing={}".format(
                self.__class__.__name__,
                self.weighted_loss,
                self.ignore_idx,
                self.label_smoothing,
            )
        )

        if self.aux_wt > 0:
            repr_str += "\n\taux_wt={}".format(self.aux_wt)
        return repr_str + "\n)"

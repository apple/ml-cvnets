#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch


class ConfusionMatrix(object):
    """
    Computes the confusion matrix and is based on `FCN <https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py>`_
    """

    def __init__(self):
        self.confusion_mat = None

    def update(self, ground_truth, prediction, n_classes):
        if self.confusion_mat is None:
            self.confusion_mat = torch.zeros(
                (n_classes, n_classes), dtype=torch.int64, device=ground_truth.device
            )
        with torch.no_grad():
            k = (ground_truth >= 0) & (ground_truth < n_classes)
            inds = n_classes * ground_truth[k].to(torch.int64) + prediction[k]
            self.confusion_mat += torch.bincount(
                inds, minlength=n_classes ** 2
            ).reshape(n_classes, n_classes)

    def reset(self):
        if self.confusion_mat is not None:
            self.confusion_mat.zero_()

    def compute(self):
        if self.confusion_mat is None:
            print("Confusion matrix is None. Check code")
            return None
        h = self.confusion_mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        diag_h = torch.diag(h)
        acc = diag_h / h.sum(1)
        iu = diag_h / (h.sum(1) + h.sum(0) - diag_h)
        return acc_global, acc, iu

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os.path
from torch.nn import functional as F
import torch
from torch import nn, Tensor
import argparse

from utils import logger
from utils.tensor_utils import tensor_to_python_float
from utils.ddp_utils import is_master
from utils.checkpoint_utils import load_state_dict
from cvnets.models.classification import build_classification_model

from . import register_distillation_loss_fn
from .. import BaseCriteria


@register_distillation_loss_fn(name="vanilla")
class VanillaDistillationLoss(BaseCriteria):
    """
        Distillation loss
    """
    def __init__(self, opts):
        alpha = getattr(opts, "loss.distillation.vanilla_alpha", 0.5)
        steps = getattr(opts, "loss.distillation.vanilla_accum_iterations", 10000)
        adaptive_wt_balance = getattr(opts, "loss.distillation.vanilla_adaptive_weight_balance", False)
        tau = getattr(opts, "loss.distillation.vanilla_tau", 1.0)
        is_distributed = getattr(opts, "ddp.use_distributed", False)
        n_gpus = getattr(opts, "dev.num_gpus", 0)
        update_freq = getattr(opts, "loss.distillation.vanilla_weight_update_freq", 10)
        distill_type = getattr(opts, "loss.distillation.vanilla_distillation_type", "soft")
        super(VanillaDistillationLoss, self).__init__()

        self.teacher_model = self.build_teacher_model(opts=opts)
        if not is_distributed and n_gpus > 0:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)
        self.teacher_model.eval()

        self.label_loss = self.build_label_loss_fn(opts=opts)
        self.tau = tau

        self.weight_label = 1.0 if adaptive_wt_balance else alpha
        self.weight_dist = 1.0 if adaptive_wt_balance else 1.0 - alpha

        self.adaptive = adaptive_wt_balance
        self.loss_acc_label = 0.0
        self.loss_acc_dist = 0.0
        self.steps = steps
        self.step_counter = 0
        self.is_distributed = is_distributed
        self.update_freq = update_freq
        self.is_master_node = is_master(opts)

        self.distillation_loss_fn = self.compute_soft_distillation_loss
        if distill_type == "hard":
            self.distillation_loss_fn = self.compute_hard_distillation_loss

    @staticmethod
    def build_teacher_model(opts) -> nn.Module:
        teacher_model_name = getattr(opts, "loss.distillation.vanilla_teacher_model", "resnet_50").lower()
        def_model_name = getattr(opts, "model.classification.name", None)
        setattr(opts, "model.classification.name", teacher_model_name)
        teacher_model = build_classification_model(opts=opts)
        setattr(opts, "model.classification.name", def_model_name)

        pretrained_wts_path = getattr(opts, "loss.distillation.vanilla_teacher_model_weights", None)

        if pretrained_wts_path is not None and os.path.isfile(pretrained_wts_path):
            pretrained_wts_dict = torch.load(pretrained_wts_path, map_location="cpu")
            load_state_dict(model=teacher_model, state_dict=pretrained_wts_dict)
        else:
            raise RuntimeError("Pretrained weights are required for teacher model ({}) in the distillation loss".format(teacher_model_name))

        return teacher_model

    @staticmethod
    def build_label_loss_fn(opts) -> BaseCriteria:
        label_loss_name = getattr(opts, "loss.distillation.vanilla_label_loss", "cross_entropy")
        if label_loss_name == "cross_entropy":
            from loss_fn.classification import ClassificationLoss
            loss_fn_name = getattr(opts, "loss.classification.name", "cross_entropy")
            setattr(opts, "loss.classification.name", label_loss_name)
            label_loss = ClassificationLoss(opts)
            setattr(opts, "loss.classification.name", loss_fn_name)
            return label_loss
        else:
            raise NotImplementedError

    def compute_soft_distillation_loss(self, input_sample, prediction) -> Tensor:
        """
            Details about soft-distillation here: https://arxiv.org/abs/2012.12877
        """

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_sample)

        multiplier = (self.tau * self.tau * 1.0) / prediction.numel()
        pred_dist = F.log_softmax(prediction / self.tau, dim=1)
        teach_dist = F.log_softmax(teacher_outputs / self.tau, dim=1)
        distillation_loss = F.kl_div(pred_dist, teach_dist, reduction="sum", log_target=True) * multiplier
        return distillation_loss

    def compute_hard_distillation_loss(self, input_sample, prediction) -> Tensor:
        """
            Details about Distillation here: https://arxiv.org/abs/1503.02531
        """
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_sample)
            teacher_labels = teacher_logits.argmax(dim=-1)

        distillation_loss = self.label_loss(input_sample=input_sample, prediction=prediction, target=teacher_labels)
        return distillation_loss

    def compute_weights(self):
        prev_wt_dist = self.weight_dist
        self.weight_dist = round(self.loss_acc_label / (self.loss_acc_dist + self.eps), 3)
        # self.loss_acc_label = 0.0
        # self.loss_acc_dist = 0.0
        if self.is_master_node:
            logger.log("{} Contribution of distillation loss w.r.t label loss is updated".format(self.__class__.__name__))
            print("\t\t Dist. loss contribution: {} -> {}".format(prev_wt_dist, self.weight_dist))

    def forward(self,  input_sample: Tensor, prediction: Tensor, target: Tensor) -> Tensor:
        distillation_loss = self.distillation_loss_fn(input_sample=input_sample, prediction=prediction)
        label_loss = self.label_loss(input_sample=input_sample, prediction=prediction, target=target)
        if self.adaptive and self.step_counter < self.steps:
            self.loss_acc_dist += tensor_to_python_float(distillation_loss, is_distributed=self.is_distributed)
            self.loss_acc_label += tensor_to_python_float(label_loss, is_distributed=self.is_distributed)

            # update the weights
            if (self.step_counter + 1) % self.update_freq == 0:
                self.compute_weights()

            self.step_counter += 1
        elif self.adaptive and self.step_counter == self.steps:
            self.compute_weights()
            self.step_counter += 1

        total_loss = (self.weight_label * label_loss) + (self.weight_dist * distillation_loss)

        return total_loss

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--loss.distillation.vanilla-teacher-model", type=str, default="resnet_50",
                           help="Teacher model for distillation")
        group.add_argument("--loss.distillation.vanilla-label-loss", type=str, default="cross_entropy", help="Label loss")
        group.add_argument("--loss.distillation.vanilla-alpha", type=float, default=0.5, help="Contribution of label loss")
        group.add_argument("--loss.distillation.vanilla-tau", type=float, default=1.0, help="Tau value")
        group.add_argument("--loss.distillation.vanilla-adaptive-weight-balance", action="store_true",
                           help="Adaptively balance the contribution of label and distribution loss")
        group.add_argument("--loss.distillation.vanilla-accum-iterations", type=int, default=10000,
                           help="Number of iterations for accumulating loss")
        group.add_argument("--loss.distillation.vanilla-weight-update-freq", type=int, default=100,
                           help="Update freq for updating the weights for label and distribution loss")
        group.add_argument("--loss.distillation.vanilla-teacher-model-weights", type=str,
                           help="Teacher model path")
        group.add_argument("--loss.distillation.vanilla-distillation-type", type=str, default="soft",
                           help="Use hard or soft labels from teacher")
        return parser

    def __repr__(self):
        label_loss_str = str(self.label_loss.__repr__()).replace("\n", "").replace("\t", "")
        return "{}(\n\t label_loss_fn={} \n\t teacher_model={} \n\t alpha={} \n\t tau={} \n\t adaptive={} \n\t accum_iter={} \n\t update_freq={} \n)".format(
            self.__class__.__name__,
            label_loss_str,
            self.teacher_model.__class__.__name__,
            self.weight_label,
            self.tau,
            self.adaptive,
            self.steps,
            self.update_freq
        )
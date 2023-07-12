#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.


import pytest
import torch

from cvnets.layers import LinearLayer
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from loss_fn.distillation.hard_distillation import HardDistillationLoss
from loss_fn.distillation.soft_kl_distillation import SoftKLLoss
from tests.configs import get_config
from tests.test_utils import unset_pretrained_models_from_opts


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("temperature", [0.1])
def test_soft_kl_loss_in_out(batch_size: int, temperature: float) -> None:
    # These tests check the input and output formats are correct or not.

    # get configuration
    config_file = "config/distillation/teacher_resnet101_student_mobilenet_v1.yaml"
    opts = get_config(config_file=config_file)
    setattr(opts, "loss.distillation.soft_kl_loss.temperature", temperature)
    unset_pretrained_models_from_opts(opts)

    criteria = SoftKLLoss(opts)

    num_classes = num_classes_from_teacher(criteria.teacher)

    input_sample = torch.randn(size=(batch_size, 3, 32, 32))

    # two cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dict with logits keys
    pred_case_1 = torch.randn(size=(batch_size, num_classes))
    pred_case_2 = {"logits": torch.randn(size=(batch_size, num_classes))}
    target = None

    for pred in [pred_case_1, pred_case_2]:
        loss_val = criteria(input_sample, pred, target)
        assert isinstance(
            loss_val, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss_val.dim() == 0, "Loss value should be a scalar"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("top_k", [1, 5])
@pytest.mark.parametrize("label_smoothing", [0.1, 0.5])
def test_hard_distillation_loss_in_out(
    batch_size: int,
    top_k: int,
    label_smoothing: float,
) -> None:
    # These tests check the input and output formats are correct or not.

    # get configuration
    config_file = "config/distillation/teacher_resnet101_student_mobilenet_v1.yaml"
    opts = get_config(config_file=config_file)
    setattr(opts, "loss.distillation.hard_distillation.topk", top_k)
    setattr(
        opts, "loss.distillation.hard_distillation.label_smoothing", label_smoothing
    )

    unset_pretrained_models_from_opts(opts)

    criteria = HardDistillationLoss(opts)

    num_classes = num_classes_from_teacher(criteria.teacher)

    input_sample = torch.randn(size=(batch_size, 3, 32, 32))

    # two cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dict with logits keys
    pred_case_1 = torch.randn(size=(batch_size, num_classes))
    pred_case_2 = {"logits": torch.randn(size=(batch_size, num_classes))}
    target = None

    for pred in [pred_case_1, pred_case_2]:
        loss_val = criteria(input_sample, pred, target)
        assert isinstance(
            loss_val, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss_val.dim() == 0, "Loss value should be a scalar"


def num_classes_from_teacher(teacher: BaseImageEncoder):
    teacher_classifier = teacher.classifier
    if isinstance(teacher_classifier, (torch.nn.Linear, LinearLayer)):
        return teacher_classifier.out_features
    elif isinstance(teacher_classifier, torch.nn.Sequential):
        last_layer = teacher_classifier[-1]
        assert isinstance(last_layer, (torch.nn.Linear, LinearLayer))
        return last_layer.out_features
    else:
        raise NotImplementedError

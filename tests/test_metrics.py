#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import math
import sys

sys.path.append("..")

import torch
import numpy as np

from metrics.metric_monitor import (
    gather_top_k_metrics,
    gather_loss,
    gather_psnr_metrics,
    gather_grad_norm,
    gather_iou_metrics,
)


def test_gather_top_k_metrics():
    # Our metrics for top_k support two input formats: Dicts and Tensors. Therefore, we can have
    # four possible cases:
    #  1. Both prediction and target are tensors
    #  2. Prediction is dict (one or more of its keys have logits) and target is tensor
    #  3. Prediction and target are dicts (both of them have atleast one same key)
    #  4. Prediction is a tensor and target is a dict (with atleast one key having target label)

    # Test for case 1
    inp_tensor = torch.tensor(
        [
            [0.02, 0.1, 0.9, 0.05, 0.02, 0.01, 0],
            [0.9, 0.05, 0.05, 0, 0, 0, 0],
            [0.4, 0.5, 0.05, 0.05, 0, 0, 0],
        ],
        dtype=torch.float,
    )

    label_tensor = torch.tensor([3, 2, 1], dtype=torch.long)

    top1_acc, top5_acc = gather_top_k_metrics(
        prediction=inp_tensor, target=label_tensor, is_distributed=False
    )
    top1_acc = round(top1_acc, 2)
    top5_acc = round(top5_acc, 2)

    np.testing.assert_almost_equal(top1_acc, 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc, 100.00, decimal=2)

    # Test for case 2
    top1_acc_dict, top5_acc_dict = gather_top_k_metrics(
        prediction={"dummy_tensor": inp_tensor},
        target=label_tensor,
        is_distributed=False,
    )
    np.testing.assert_almost_equal(top1_acc_dict["dummy_tensor"], 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc_dict["dummy_tensor"], 100.00, decimal=2)

    # Test for case 3
    # We need atleast one common key that matches the constraints for computing top-1 accuracy.
    # i.e., predicted tensor has shape [Batch, Num_classes] and label tensor has shape [Batch]
    top1_acc_dict, top5_acc_dict = gather_top_k_metrics(
        prediction={"classification": inp_tensor},
        target={"classification": label_tensor},
        is_distributed=False,
    )
    np.testing.assert_almost_equal(top1_acc_dict["classification"], 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc_dict["classification"], 100.00, decimal=2)

    top1_acc_dict, top5_acc_dict = gather_top_k_metrics(
        prediction={"classification": inp_tensor},
        target={"classification": label_tensor, "classification_dummy": label_tensor},
        is_distributed=False,
    )
    keys_in_prediction = list(top1_acc_dict.keys())
    np.testing.assert_almost_equal(len(keys_in_prediction), 1)
    np.testing.assert_equal(keys_in_prediction[0], "classification")

    np.testing.assert_almost_equal(top1_acc_dict["classification"], 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc_dict["classification"], 100.00, decimal=2)

    # test for case 4
    top1_acc_dict, top5_acc_dict = gather_top_k_metrics(
        prediction=inp_tensor,
        target={"dummy_label": label_tensor},
        is_distributed=False,
    )
    np.testing.assert_almost_equal(top1_acc_dict["dummy_label"], 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc_dict["dummy_label"], 100.00, decimal=2)


def test_gather_loss():
    # loss could be a Tensor or Dictionary
    loss = torch.tensor([3.2], dtype=torch.float)

    out = gather_loss(loss, is_distributed=False)
    np.testing.assert_almost_equal(out, 3.2)

    loss_dict = {
        "aux_loss": torch.tensor([4.2]),
        "total_loss": torch.tensor(
            [5.2]
        ),  # total loss key is mandatory for a loss in dict format
    }
    out_dict = gather_loss(loss_dict, is_distributed=False)

    out_dict_keys = list(out_dict.keys())

    np.testing.assert_equal(len(out_dict_keys), 2)
    np.testing.assert_equal(out_dict_keys[0], "aux_loss")
    np.testing.assert_equal(out_dict_keys[1], "total_loss")
    np.testing.assert_almost_equal(out_dict["aux_loss"], 4.2, decimal=2)
    np.testing.assert_almost_equal(out_dict["total_loss"], 5.2, decimal=2)


def test_gather_psnr_metrics():
    # Our metrics for PSNR support two input formats: Dicts and Tensors. Therefore, we can have
    # four possible cases:
    #  1. Both prediction and target are tensors
    #  2. Prediction is dict (one or more of its keys have logits) and target is tensor
    #  3. Prediction and target are dicts (both of them have atleast one same key)
    #  4. Prediction is a tensor and target is a dict (with atleast one key having target label)

    # Test for case 1
    inp_tensor = torch.randn((3, 2), dtype=torch.float)
    target_tensor = inp_tensor

    # Ideally, the PSNR should be infinite when input and target are the same, because error between
    # signal and noise is 0. However, we add a small eps value (error of 1e-10) in the computation
    # for numerical stability. Therefore, PSNR will not be infinite.
    expected_psnr = 10.0 * math.log10(255.0**2 / 1e-10)

    psnr = gather_psnr_metrics(
        prediction=inp_tensor, target=target_tensor, is_distributed=False
    )

    np.testing.assert_almost_equal(psnr, expected_psnr, decimal=2)

    # Test for case 2
    psnr_dict = gather_psnr_metrics(
        prediction={"dummy_tensor": inp_tensor},
        target=target_tensor,
        is_distributed=False,
    )
    np.testing.assert_almost_equal(psnr_dict["dummy_tensor"], expected_psnr, decimal=2)

    # Test for case 3
    # We need atleast one common key that matches the constraints for computing top-1 accuracy.
    # i.e., predicted tensor has shape [Batch, Num_classes] and label tensor has shape [Batch]
    psnr_dict = gather_psnr_metrics(
        prediction={"classification": inp_tensor},
        target={"classification": target_tensor},
        is_distributed=False,
    )
    np.testing.assert_almost_equal(
        psnr_dict["classification"], expected_psnr, decimal=2
    )

    psnr_dict = gather_psnr_metrics(
        prediction={"classification": inp_tensor},
        target={"classification": target_tensor, "classification_dummy": target_tensor},
        is_distributed=False,
    )
    keys_in_prediction = list(psnr_dict.keys())
    np.testing.assert_almost_equal(len(keys_in_prediction), 1)
    np.testing.assert_equal(keys_in_prediction[0], "classification")

    np.testing.assert_almost_equal(
        psnr_dict["classification"], expected_psnr, decimal=2
    )

    # test for case 4
    psnr_dict = gather_psnr_metrics(
        prediction=inp_tensor,
        target={"dummy_label": target_tensor},
        is_distributed=False,
    )
    np.testing.assert_almost_equal(psnr_dict["dummy_label"], expected_psnr, decimal=2)


def test_gather_grad_norm():
    # Grad norm could be a Tensor or Dictionary
    grad_norm = torch.tensor([3.2], dtype=torch.float)

    out = gather_grad_norm(grad_norm, is_distributed=False)
    np.testing.assert_almost_equal(out, 3.2)

    grad_norm_dict = {
        "dummy_norm_a": torch.tensor([4.2]),
        "dummy_norm_b": torch.tensor([5.2]),
    }
    out_dict = gather_grad_norm(grad_norm_dict, is_distributed=False)

    out_dict_keys = list(out_dict.keys())

    np.testing.assert_equal(len(out_dict_keys), 2)
    np.testing.assert_equal(out_dict_keys[0], "dummy_norm_a")
    np.testing.assert_equal(out_dict_keys[1], "dummy_norm_b")
    np.testing.assert_almost_equal(out_dict["dummy_norm_a"], 4.2, decimal=2)
    np.testing.assert_almost_equal(out_dict["dummy_norm_b"], 5.2, decimal=2)


def test_gather_iou_metrics():
    # currently we only support tensor only

    # [Batch, num_classes, height, width]
    # in this example, [1, 2, 2, 3]
    prediction = torch.tensor(
        [
            [
                [[0.2, 0.8, 0.2], [0.9, 0.2, 0.1]],
                [[0.8, 0.2, 0.8], [0.1, 0.8, 0.9]],  # spatial dms
            ]  # classes
        ]  # batch
    )

    target = torch.tensor([[[0, 0, 0], [0, 1, 1]]])

    expected_inter = np.array([2.0, 2.0])
    expected_union = np.array([4.0, 4.0])

    inter, union = gather_iou_metrics(prediction, target, is_distributed=False)

    np.testing.assert_equal(actual=inter, desired=expected_inter)
    np.testing.assert_equal(actual=union, desired=expected_union)

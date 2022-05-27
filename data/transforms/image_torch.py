#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import math
from typing import Dict
import argparse
import torch
from torchvision.transforms import functional as F
from torch.nn import functional as F_torch

from utils import logger

from . import register_transformations, BaseTransformation


# Copied from PyTorch Torchvision
@register_transformations(name="random_mixup", type="image_torch")
class RandomMixup(BaseTransformation):
    """
    Given a batch of input images and labels, this class randomly applies the
    `Mixup transformation <https://arxiv.org/abs/1710.09412>`_

    Args:
        num_classes (int): Number of classes in the dataset
    """

    def __init__(self, opts, num_classes: int, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        alpha = getattr(opts, "image_augmentation.mixup.alpha", 1.0)
        assert (
            num_classes > 0
        ), "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = getattr(opts, "image_augmentation.mixup.p", 0.5)
        self.alpha = alpha
        self.inplace = getattr(opts, "image_augmentation.mixup.inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.mixup.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.mixup.alpha",
            type=float,
            default=1.0,
            help="Alpha for MixUp augmentation. Defaults to 1.0",
        )
        group.add_argument(
            "--image-augmentation.mixup.p",
            type=float,
            default=0.5,
            help="Probability for applying mixup augmentation. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.mixup.inplace",
            action="store_true",
            default=False,
            help="Apply Mixup augmentation inplace. Defaults to False.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if torch.rand(1).item() >= self.p:
            return data

        image_tensor, target_tensor = data.pop("image"), data.pop("label")

        if image_tensor.ndim != 4:
            logger.error(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            logger.error(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            logger.error(
                f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
            )
        if target_tensor.dtype != torch.int64:
            logger.error(
                f"Target dtype should be torch.int64. Got {target_tensor.dtype}"
            )

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        image_tensor.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)

        data["image"] = image_tensor
        data["label"] = target_tensor

        return data

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )


@register_transformations(name="random_cutmix", type="image_torch")
class RandomCutmix(BaseTransformation):
    """
    Given a batch of input images and labels, this class randomly applies the
    `CutMix transformation <https://arxiv.org/abs/1905.04899>`_

    Args:
        num_classes (int): Number of classes in the dataset
    """

    def __init__(self, opts, num_classes: int, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        alpha = getattr(opts, "image_augmentation.cutmix.alpha", 1.0)
        assert (
            num_classes > 0
        ), "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = getattr(opts, "image_augmentation.cutmix.p", 0.5)
        self.alpha = alpha
        self.inplace = getattr(opts, "image_augmentation.cutmix.inplace", False)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.cutmix.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )

        group.add_argument(
            "--image-augmentation.cutmix.alpha",
            type=float,
            default=1.0,
            help="Alpha for cutmix augmentation. Defaults to 1.0",
        )
        group.add_argument(
            "--image-augmentation.cutmix.p",
            type=float,
            default=0.5,
            help="Probability for applying cutmix augmentation. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.cutmix.inplace",
            action="store_true",
            default=False,
            help="Apply cutmix operation inplace. Defaults to False",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if torch.rand(1).item() >= self.p:
            return data

        image_tensor, target_tensor = data.pop("image"), data.pop("label")

        if image_tensor.ndim != 4:
            logger.error(f"Batch ndim should be 4. Got {image_tensor.ndim}")
        if target_tensor.ndim != 1:
            logger.error(f"Target ndim should be 1. Got {target_tensor.ndim}")
        if not image_tensor.is_floating_point():
            logger.error(
                f"Batch dtype should be a float tensor. Got {image_tensor.dtype}."
            )
        if target_tensor.dtype != torch.int64:
            logger.error(
                f"Target dtype should be torch.int64. Got {target_tensor.dtype}"
            )

        if not self.inplace:
            image_tensor = image_tensor.clone()
            target_tensor = target_tensor.clone()

        if target_tensor.ndim == 1:
            target_tensor = F_torch.one_hot(
                target_tensor, num_classes=self.num_classes
            ).to(dtype=image_tensor.dtype)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = image_tensor.roll(1, 0)
        target_rolled = target_tensor.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        W, H = F.get_image_size(image_tensor)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        image_tensor[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target_tensor.mul_(lambda_param).add_(target_rolled)

        data["image"] = image_tensor
        data["label"] = target_tensor

        return data

    def __repr__(self) -> str:
        return "{}(num_classes={}, p={}, alpha={}, inplace={})".format(
            self.__class__.__name__, self.num_classes, self.p, self.alpha, self.inplace
        )

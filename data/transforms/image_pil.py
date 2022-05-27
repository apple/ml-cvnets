#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
from PIL import Image, ImageFilter
from utils import logger
import numpy as np
import random
import torch
import math
import argparse
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Sequence, Dict, Any, Union, Tuple, List, Optional

from . import register_transformations, BaseTransformation
from .utils import jaccard_numpy, setup_size

INTERPOLATION_MODE_MAP = {
    "nearest": T.InterpolationMode.NEAREST,
    "bilinear": T.InterpolationMode.BILINEAR,
    "bicubic": T.InterpolationMode.BICUBIC,
    "cubic": T.InterpolationMode.BICUBIC,
    "box": T.InterpolationMode.BOX,
    "hamming": T.InterpolationMode.HAMMING,
    "lanczos": T.InterpolationMode.LANCZOS,
}


def _interpolation_modes_from_str(name: str) -> T.InterpolationMode:
    return INTERPOLATION_MODE_MAP[name]


def _crop_fn(data: Dict, top: int, left: int, height: int, width: int) -> Dict:
    """Helper function for cropping"""
    img = data["image"]
    data["image"] = F.crop(img, top=top, left=left, height=height, width=width)

    if "mask" in data:
        mask = data.pop("mask")
        data["mask"] = F.crop(mask, top=top, left=left, height=height, width=width)

    if "box_coordinates" in data:
        boxes = data.pop("box_coordinates")

        area_before_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )

        boxes[..., 0::2] = np.clip(boxes[..., 0::2] - left, a_min=0, a_max=left + width)
        boxes[..., 1::2] = np.clip(boxes[..., 1::2] - top, a_min=0, a_max=top + height)

        area_after_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )
        area_ratio = area_after_cropping / (area_before_cropping + 1)

        # keep the boxes whose area is atleast 20% of the area before cropping
        keep = area_ratio >= 0.2

        box_labels = data.pop("box_labels")

        data["box_coordinates"] = boxes[keep]
        data["box_labels"] = box_labels[keep]

    if "instance_mask" in data:
        assert "instance_coords" in data

        instance_masks = data.pop("instance_mask")
        data["instance_mask"] = F.crop(
            instance_masks, top=top, left=left, height=height, width=width
        )

        instance_coords = data.pop("instance_coords")
        instance_coords[..., 0::2] = np.clip(
            instance_coords[..., 0::2] - left, a_min=0, a_max=left + width
        )
        instance_coords[..., 1::2] = np.clip(
            instance_coords[..., 1::2] - top, a_min=0, a_max=top + height
        )
        data["instance_coords"] = instance_coords

    return data


def _resize_fn(
    data: Dict,
    size: Union[Sequence, int],
    interpolation: Optional[T.InterpolationMode or str] = T.InterpolationMode.BILINEAR,
) -> Dict:
    """Helper function for resizing"""
    img = data["image"]

    w, h = F.get_image_size(img)

    if isinstance(size, Sequence) and len(size) == 2:
        size_h, size_w = size[0], size[1]
    elif isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return data

        if w < h:
            size_h = int(size * h / w)

            size_w = size
        else:
            size_w = int(size * w / h)
            size_h = size
    else:
        raise TypeError(
            "Supported size args are int or tuple of length 2. Got inappropriate size arg: {}".format(
                size
            )
        )

    if isinstance(interpolation, str):
        interpolation = _interpolation_modes_from_str(name=interpolation)

    data["image"] = F.resize(
        img=img, size=[size_h, size_w], interpolation=interpolation
    )

    if "mask" in data:
        mask = data.pop("mask")
        resized_mask = F.resize(
            img=mask, size=[size_h, size_w], interpolation=T.InterpolationMode.NEAREST
        )
        data["mask"] = resized_mask

    if "box_coordinates" in data:
        boxes = data.pop("box_coordinates")
        boxes[:, 0::2] *= 1.0 * size_w / w
        boxes[:, 1::2] *= 1.0 * size_h / h
        data["box_coordinates"] = boxes

    if "instance_mask" in data:
        assert "instance_coords" in data

        instance_masks = data.pop("instance_mask")

        resized_instance_masks = F.resize(
            img=instance_masks,
            size=[size_h, size_w],
            interpolation=T.InterpolationMode.NEAREST,
        )
        data["instance_mask"] = resized_instance_masks

        instance_coords = data.pop("instance_coords")
        instance_coords = instance_coords.astype(np.float)
        instance_coords[..., 0::2] *= 1.0 * size_w / w
        instance_coords[..., 1::2] *= 1.0 * size_h / h
        data["instance_coords"] = instance_coords

    return data


@register_transformations(name="random_resized_crop", type="image_pil")
class RandomResizedCrop(BaseTransformation, T.RandomResizedCrop):
    """
    This class crops a random portion of an image and resize it to a given size.
    """

    def __init__(self, opts, size: Union[Sequence, int], *args, **kwargs) -> None:
        interpolation = getattr(
            opts, "image_augmentation.random_resized_crop.interpolation", "bilinear"
        )
        scale = getattr(
            opts, "image_augmentation.random_resized_crop.scale", (0.08, 1.0)
        )
        ratio = getattr(
            opts,
            "image_augmentation.random_resized_crop.aspect_ratio",
            (3.0 / 4.0, 4.0 / 3.0),
        )

        BaseTransformation.__init__(self, opts=opts)

        T.RandomResizedCrop.__init__(
            self, size=size, scale=scale, ratio=ratio, interpolation=interpolation
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.random-resized-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.interpolation",
            type=str,
            default="bilinear",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Interpolation method for resizing. Defaults to bilinear.",
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.scale",
            type=tuple,
            default=(0.08, 1.0),
            help="Specifies the lower and upper bounds for the random area of the crop, before resizing."
            " The scale is defined with respect to the area of the original image. Defaults to "
            "(0.08, 1.0)",
        )
        group.add_argument(
            "--image-augmentation.random-resized-crop.aspect-ratio",
            type=float or tuple,
            default=(3.0 / 4.0, 4.0 / 3.0),
            help="lower and upper bounds for the random aspect ratio of the crop, before resizing. "
            "Defaults to (3./4., 4./3.)",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        img = data["image"]
        i, j, h, w = super().get_params(img=img, scale=self.scale, ratio=self.ratio)
        data = _crop_fn(data=data, top=i, left=j, height=h, width=w)
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(scale={}, ratio={}, size={}, interpolation={})".format(
            self.__class__.__name__,
            self.scale,
            self.ratio,
            self.size,
            self.interpolation,
        )


@register_transformations(name="auto_augment", type="image_pil")
class AutoAugment(BaseTransformation, T.AutoAugment):
    """
    This class implements the `AutoAugment data augmentation <https://arxiv.org/pdf/1805.09501.pdf>`_ method.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        policy_name = getattr(
            opts, "image_augmentation.auto_augment.policy", "imagenet"
        )
        interpolation = getattr(
            opts, "image_augmentation.auto_augment.interpolation", "bilinear"
        )
        if policy_name == "imagenet":
            policy = T.AutoAugmentPolicy.IMAGENET
        else:
            raise NotImplemented

        if isinstance(interpolation, str):
            interpolation = _interpolation_modes_from_str(name=interpolation)

        BaseTransformation.__init__(self, opts=opts)
        T.AutoAugment.__init__(self, policy=policy, interpolation=interpolation)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.auto-augment.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.auto-augment.policy",
            type=str,
            default="imagenet",
            help="Auto-augment policy name. Defaults to imagenet.",
        )
        group.add_argument(
            "--image-augmentation.auto-augment.interpolation",
            type=str,
            default="bilinear",
            help="Auto-augment interpolation method. Defaults to bilinear interpolation",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if "box_coordinates" in data or "mask" in data or "instance_masks" in data:
            logger.error(
                "{} is only supported for classification tasks".format(
                    self.__class__.__name__
                )
            )

        img = data["image"]
        img = super().forward(img)
        data["image"] = img
        return data

    def __repr__(self) -> str:
        return "{}(policy={}, interpolation={})".format(
            self.__class__.__name__, self.policy, self.interpolation
        )


@register_transformations(name="rand_augment", type="image_pil")
class RandAugment(BaseTransformation, T.RandAugment):
    """
    This class implements the `RandAugment data augmentation <https://arxiv.org/abs/1909.13719>`_ method.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_ops = getattr(opts, "image_augmentation.rand_augment.num_ops", 2)
        magnitude = getattr(opts, "image_augmentation.rand_augment.magnitude", 9)
        num_magnitude_bins = getattr(
            opts, "image_augmentation.rand_augment.num_magnitude_bins", 31
        )
        interpolation = getattr(
            opts, "image_augmentation.rand_augment.interpolation", "bilinear"
        )

        BaseTransformation.__init__(self, opts=opts)

        if isinstance(interpolation, str):
            interpolation = _interpolation_modes_from_str(name=interpolation)

        T.RandAugment.__init__(
            self,
            num_ops=num_ops,
            magnitude=magnitude,
            num_magnitude_bins=num_magnitude_bins,
            interpolation=interpolation,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.rand-augment.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.rand-augment.num-ops",
            type=int,
            default=2,
            help="Number of augmentation transformations to apply sequentially. Defaults to 2.",
        )
        group.add_argument(
            "--image-augmentation.rand-augment.magnitude",
            type=int,
            default=9,
            help="Magnitude for all the transformations. Defaults to 9",
        )
        group.add_argument(
            "--image-augmentation.rand-augment.num-magnitude-bins",
            type=int,
            default=31,
            help="The number of different magnitude values. Defaults to 31.",
        )
        group.add_argument(
            "--image-augmentation.rand-augment.interpolation",
            type=str,
            default="bilinear",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Desired interpolation method. Defaults to bilinear",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if "box_coordinates" in data or "mask" in data or "instance_masks" in data:
            logger.error(
                "{} is only supported for classification tasks".format(
                    self.__class__.__name__
                )
            )

        img = data["image"]
        img = super().forward(img)
        data["image"] = img
        return data

    def __repr__(self) -> str:
        return "{}(num_ops={}, magnitude={}, num_magnitude_bins={}, interpolation={})".format(
            self.__class__.__name__,
            self.num_ops,
            self.magnitude,
            self.num_magnitude_bins,
            self.interpolation,
        )


@register_transformations(name="random_horizontal_flip", type="image_pil")
class RandomHorizontalFlip(BaseTransformation):
    """
    This class implements random horizontal flipping method
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        p = getattr(opts, "image_augmentation.random_horizontal_flip.p", 0.5)
        super().__init__(opts=opts)
        self.p = p

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.random-horizontal-flip.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-horizontal-flip.p",
            type=float,
            default=0.5,
            help="Probability for applying random horizontal flip",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if random.random() <= self.p:
            img = data["image"]
            width, height = F.get_image_size(img)
            data["image"] = F.hflip(img)

            if "mask" in data:
                mask = data.pop("mask")
                data["mask"] = F.hflip(mask)

            if "box_coordinates" in data:
                boxes = data.pop("box_coordinates")
                boxes[..., 0::2] = width - boxes[..., 2::-2]
                data["box_coordinates"] = boxes

            if "instance_mask" in data:
                assert "instance_coords" in data

                instance_coords = data.pop("instance_coords")
                instance_coords[..., 0::2] = width - instance_coords[..., 2::-2]
                data["instance_coords"] = instance_coords

                instance_masks = data.pop("instance_mask")
                data["instance_mask"] = F.hflip(instance_masks)
        return data

    def __repr__(self) -> str:
        return "{}(p={})".format(self.__class__.__name__, self.p)


@register_transformations(name="random_rotate", type="image_pil")
class RandomRotate(BaseTransformation):
    """
    This class implements random rotation method
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.angle = getattr(opts, "image_augmentation.random_rotate.angle", 10)
        self.mask_fill = getattr(opts, "image_augmentation.random_rotate.mask_fill", 0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.random-rotate.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-rotate.angle",
            type=float,
            default=10,
            help="Angle for rotation. Defaults to 10. The angle is sampled "
            "uniformly from [-angle, angle]",
        )
        group.add_argument(
            "--image-augmentation.random-rotate.mask-fill",
            default=0,
            help="Fill value for the segmentation mask. Defaults to 0.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:

        data_keys = list(data.keys())
        if "box_coordinates" in data_keys or "instance_mask" in data_keys:
            logger.error("{} supports only images and masks")

        rand_angle = random.uniform(-self.angle, self.angle)
        img = data.pop("image")
        data["image"] = F.rotate(
            img, angle=rand_angle, interpolation=F.InterpolationMode.BILINEAR, fill=0
        )
        if "mask" in data:
            mask = data.pop("mask")
            data["mask"] = F.rotate(
                mask,
                angle=rand_angle,
                interpolation=F.InterpolationMode.NEAREST,
                fill=self.mask_fill,
            )
        return data

    def __repr__(self) -> str:
        return "{}(angle={}, mask_fill={})".format(
            self.__class__.__name__, self.angle, self.mask_fill
        )


@register_transformations(name="resize", type="image_pil")
class Resize(BaseTransformation):
    """
    This class implements resizing operation.

    .. note::
    Two possible modes for resizing.
    1. Resize while maintaining aspect ratio. To enable this option, pass int as a size
    2. Resize to a fixed size. To enable this option, pass a tuple of height and width as a size

    .. note::
        If img_size is passed as a positional argument, then it will override size from args
    """

    def __init__(
        self,
        opts,
        img_size: Optional[Union[Tuple[int, int], int]] = None,
        *args,
        **kwargs
    ) -> None:
        interpolation = getattr(
            opts, "image_augmentation.resize.interpolation", "bilinear"
        )
        super().__init__(opts=opts)

        # img_size argument is useful for implementing multi-scale sampler
        size = (
            getattr(opts, "image_augmentation.resize.size", None)
            if img_size is None
            else img_size
        )
        if size is None:
            logger.error("Size can not be None in {}".format(self.__class__.__name__))

        # Possible modes.
        # 1. Resize while maintaining aspect ratio. To enable this option, pass int as a size
        # 2. Resize to a fixed size. To enable this option, pass a tuple of height and width as a size

        if isinstance(size, Sequence) and len(size) > 2:
            logger.error(
                "The length of size should be either 1 or 2 in {}".format(
                    self.__class__.__name__
                )
            )

        if not (isinstance(size, Sequence) or isinstance(size, int)):
            logger.error(
                "Size needs to be either Tuple of length 2 or an integer in {}. Got: {}".format(
                    self.__class__.__name__, size
                )
            )

        self.size = size
        self.interpolation = interpolation
        self.maintain_aspect_ratio = True if isinstance(size, int) else False

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.resize.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.resize.interpolation",
            type=str,
            default="bilinear",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Desired interpolation method for resizing. Defaults to bilinear",
        )
        group.add_argument(
            "--image-augmentation.resize.size",
            type=int,
            nargs="+",
            default=None,
            help="Resize image to the specified size. If int is passed, then shorter side is resized"
            "to the specified size and longest side is resized while maintaining aspect ratio."
            "Defaults to None.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        return _resize_fn(data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(size={}, interpolation={}, maintain_aspect_ratio={})".format(
            self.__class__.__name__,
            self.size,
            self.interpolation,
            self.maintain_aspect_ratio,
        )


@register_transformations(name="center_crop", type="image_pil")
class CenterCrop(BaseTransformation):
    """
    This class implements center cropping method.

    .. note::
        This class assumes that the input size is greater than or equal to the desired size.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        size = getattr(opts, "image_augmentation.center_crop.size", None)

        if size is None:
            logger.error("Size cannot be None in {}".format(self.__class__.__name__))

        if isinstance(size, Sequence) and len(size) == 2:
            self.height, self.width = size[0], size[1]
        elif isinstance(size, Sequence) and len(size) == 1:
            self.height = self.width = size[0]
        elif isinstance(size, int):
            self.height = self.width = size
        else:
            logger.error("Scale should be either an int or tuple of ints")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.center-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.center-crop.size",
            type=int,
            nargs="+",
            default=None,
            help="Center crop size. Defaults to None.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        width, height = F.get_image_size(data["image"])
        i = (height - self.height) // 2
        j = (width - self.width) // 2
        return _crop_fn(data=data, top=i, left=j, height=self.height, width=self.width)

    def __repr__(self) -> str:
        return "{}(size=(h={}, w={}))".format(
            self.__class__.__name__, self.height, self.width
        )


@register_transformations(name="ssd_cropping", type="image_pil")
class SSDCroping(BaseTransformation):
    """
    This class implements cropping method for `Single shot object detector <https://arxiv.org/abs/1512.02325>`_.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)

        self.iou_sample_opts = getattr(
            opts,
            "image_augmentation.ssd_crop.iou_thresholds",
            [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        )
        self.trials = getattr(opts, "image_augmentation.ssd_crop.n_trials", 40)
        self.min_aspect_ratio = getattr(
            opts, "image_augmentation.ssd_crop.min_aspect_ratio", 0.5
        )
        self.max_aspect_ratio = getattr(
            opts, "image_augmentation.ssd_crop.max_aspect_ratio", 2.0
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.ssd-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.ssd-crop.iou-thresholds",
            type=float,
            nargs="+",
            default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            help="IoU thresholds for SSD cropping. Defaults to [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]",
        )
        group.add_argument(
            "--image-augmentation.ssd-crop.n-trials",
            type=int,
            default=40,
            help="Number of trials for SSD cropping. Defaults to 40",
        )
        group.add_argument(
            "--image-augmentation.ssd-crop.min-aspect-ratio",
            type=float,
            default=0.5,
            help="Min. aspect ratio in SSD Cropping. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.ssd-crop.max-aspect-ratio",
            type=float,
            default=2.0,
            help="Max. aspect ratio in SSD Cropping. Defaults to 2.0",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if "box_coordinates" in data:
            boxes = data["box_coordinates"]

            # guard against no boxes
            if boxes.shape[0] == 0:
                return data

            image = data["image"]
            labels = data["box_labels"]
            width, height = F.get_image_size(image)

            while True:
                # randomly choose a mode
                min_jaccard_overalp = random.choice(self.iou_sample_opts)
                if min_jaccard_overalp == 0.0:
                    return data

                for _ in range(self.trials):
                    new_w = int(random.uniform(0.3 * width, width))
                    new_h = int(random.uniform(0.3 * height, height))

                    aspect_ratio = new_h / new_w
                    if not (
                        self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
                    ):
                        continue

                    left = int(random.uniform(0, width - new_w))
                    top = int(random.uniform(0, height - new_h))

                    # convert to integer rect x1,y1,x2,y2
                    rect = np.array([left, top, left + new_w, top + new_h])

                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    ious = jaccard_numpy(boxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if ious.max() < min_jaccard_overalp:
                        continue

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # if image size is too small, try again
                    if (rect[3] - rect[1]) < 100 or (rect[2] - rect[0]) < 100:
                        continue

                    # cut the crop from the image
                    image = F.crop(image, top=top, left=left, width=new_w, height=new_h)

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # take only matching gt labels
                    current_labels = labels[mask]

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    data["image"] = image
                    data["box_labels"] = current_labels
                    data["box_coordinates"] = current_boxes

                    if "mask" in data:
                        mask = data.pop("mask")
                        data["mask"] = F.crop(
                            mask, top=top, left=left, width=new_w, height=new_h
                        )

                    if "instance_mask" in data:
                        assert "instance_coords" in data
                        instance_masks = data.pop("instance_mask")
                        data["instance_mask"] = F.crop(
                            instance_masks,
                            top=top,
                            left=left,
                            width=new_w,
                            height=new_h,
                        )

                        instance_coords = data.pop("instance_coords")
                        # should we use the box left and top corner or the crop's
                        instance_coords[..., :2] = np.maximum(
                            instance_coords[..., :2], rect[:2]
                        )
                        # adjust to crop (by substracting crop's left,top)
                        instance_coords[..., :2] -= rect[:2]

                        instance_coords[..., 2:] = np.minimum(
                            instance_coords[..., 2:], rect[2:]
                        )
                        # adjust to crop (by substracting crop's left,top)
                        instance_coords[..., 2:] -= rect[:2]
                        data["instance_coords"] = instance_coords

                    return data
        return data


@register_transformations(name="photo_metric_distort", type="image_pil")
class PhotometricDistort(BaseTransformation):
    """
    This class implements Photometeric distorion.

    .. note::
        Hyper-parameters of PhotoMetricDistort in PIL and OpenCV are different. Be careful
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        # contrast
        alpha_min = getattr(
            opts, "image_augmentation.photo_metric_distort.alpha_min", 0.5
        )
        alpha_max = getattr(
            opts, "image_augmentation.photo_metric_distort.alpha_max", 1.5
        )
        contrast = T.ColorJitter(contrast=[alpha_min, alpha_max])

        # brightness
        beta_min = getattr(
            opts, "image_augmentation.photo_metric_distort.beta_min", 0.875
        )
        beta_max = getattr(
            opts, "image_augmentation.photo_metric_distort.beta_max", 1.125
        )
        brightness = T.ColorJitter(brightness=[beta_min, beta_max])

        # saturation
        gamma_min = getattr(
            opts, "image_augmentation.photo_metric_distort.gamma_min", 0.5
        )
        gamma_max = getattr(
            opts, "image_augmentation.photo_metric_distort.gamma_max", 1.5
        )
        saturation = T.ColorJitter(saturation=[gamma_min, gamma_max])

        # Hue
        delta_min = getattr(
            opts, "image_augmentation.photo_metric_distort.delta_min", -0.05
        )
        delta_max = getattr(
            opts, "image_augmentation.photo_metric_distort.delta_max", 0.05
        )
        hue = T.ColorJitter(hue=[delta_min, delta_max])

        super().__init__(opts=opts)
        self._brightness = brightness
        self._contrast = contrast
        self._hue = hue
        self._saturation = saturation
        self.p = getattr(opts, "image_augmentation.photo_metric_distort.p", 0.5)

    def __repr__(self) -> str:
        return "{}(contrast={}, brightness={}, saturation={}, hue={})".format(
            self.__class__.__name__,
            self._contrast.contrast,
            self._brightness.brightness,
            self._saturation.saturation,
            self._hue.hue,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.alpha-min",
            type=float,
            default=0.5,
            help="Min. alpha value for contrast. Should be > 0. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.alpha-max",
            type=float,
            default=1.5,
            help="Max. alpha value for contrast. Should be > 0. Defaults to 1.5",
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.beta-min",
            type=float,
            default=0.875,
            help="Min. beta value for brightness. Should be > 0. Defaults to 0.8",
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.beta-max",
            type=float,
            default=1.125,
            help="Max. beta value for brightness. Should be > 0. Defaults to 1.2",
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.gamma-min",
            type=float,
            default=0.5,
            help="Min. gamma value for saturation. Should be > 0. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.gamma-max",
            type=float,
            default=1.5,
            help="Max. gamma value for saturation. Should be > 0. Defaults to 1.5",
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.delta-min",
            type=float,
            default=-0.05,
            help="Min. delta value for Hue. Should be between -1 and 1. Defaults to -0.05",
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.delta-max",
            type=float,
            default=0.05,
            help="Max. delta value for Hue. Should be between -1 and 1. Defaults to 0.05",
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.p",
            type=float,
            default=0.5,
            help="Probability for applying a distortion. Defaults to 0.5",
        )

        return parser

    def _apply_transformations(self, image):
        r = np.random.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < self.p
        if contrast_before and r[2] < self.p:
            image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before and r[5] < self.p:
            image = self._contrast(image)

        if r[6] < self.p and image.mode != "L":
            # Only permute channels for RGB images
            # [H, W, C] format
            image_np = np.asarray(image)
            n_channels = image_np.shape[2]
            image_np = image_np[..., np.random.permutation(range(n_channels))]
            image = Image.fromarray(image_np)
        return image

    def __call__(self, data: Dict) -> Dict:
        image = data.pop("image")
        data["image"] = self._apply_transformations(image)
        return data


@register_transformations(name="box_percent_coords", type="image_pil")
class BoxPercentCoords(BaseTransformation):
    """
    This class converts the box coordinates to percent
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)

    def __call__(self, data: Dict) -> Dict:
        if "box_coordinates" in data:
            boxes = data.pop("box_coordinates")
            image = data["image"]
            width, height = F.get_image_size(image)

            boxes = boxes.astype(np.float)

            boxes[..., 0::2] /= width
            boxes[..., 1::2] /= height
            data["box_coordinates"] = boxes

        return data


@register_transformations(name="instance_processor", type="image_pil")
class InstanceProcessor(BaseTransformation):
    """
    This class processes the instance masks.
    """

    def __init__(
        self,
        opts,
        instance_size: Optional[Union[int, Tuple[int, ...]]] = 16,
        *args,
        **kwargs
    ) -> None:
        super().__init__(opts=opts)
        self.instance_size = setup_size(instance_size)

    def __call__(self, data: Dict) -> Dict:

        if "instance_mask" in data:
            assert "instance_coords" in data
            instance_masks = data.pop("instance_mask")
            instance_coords = data.pop("instance_coords")
            instance_coords = instance_coords.astype(np.int)

            valid_boxes = (instance_coords[..., 3] > instance_coords[..., 1]) & (
                instance_coords[..., 2] > instance_coords[..., 0]
            )
            instance_masks = instance_masks[valid_boxes]
            instance_coords = instance_coords[valid_boxes]

            num_instances = instance_masks.shape[0]

            resized_instances = []
            for i in range(num_instances):
                # format is [N, H, W]
                instance_m = instance_masks[i]
                box_coords = instance_coords[i]

                instance_m = F.crop(
                    instance_m,
                    top=box_coords[1],
                    left=box_coords[0],
                    height=box_coords[3] - box_coords[1],
                    width=box_coords[2] - box_coords[0],
                )
                # need to unsqueeze and squeeze to make F.resize work
                instance_m = F.resize(
                    instance_m.unsqueeze(0),
                    size=self.instance_size,
                    interpolation=T.InterpolationMode.NEAREST,
                ).squeeze(0)
                resized_instances.append(instance_m)

            if len(resized_instances) == 0:
                resized_instances = torch.zeros(
                    size=(1, self.instance_size[0], self.instance_size[1]),
                    dtype=torch.long,
                )
                instance_coords = np.array(
                    [[0, 0, self.instance_size[0], self.instance_size[1]]]
                )
            else:
                resized_instances = torch.stack(resized_instances, dim=0)

            data["instance_mask"] = resized_instances
            data["instance_coords"] = instance_coords.astype(np.float)
        return data


@register_transformations(name="random_resize", type="image_pil")
class RandomResize(BaseTransformation):
    """
    This class implements random resizing method.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        min_ratio = getattr(opts, "image_augmentation.random_resize.min_ratio", 0.5)
        max_ratio = getattr(opts, "image_augmentation.random_resize.max_ratio", 2.0)
        interpolation = getattr(
            opts, "image_augmentation.random_resize.interpolation", "bilinear"
        )

        max_scale_long_edge = getattr(
            opts, "image_augmentation.random_resize.max_scale_long_edge", None
        )
        max_scale_short_edge = getattr(
            opts, "image_augmentation.random_resize.max_scale_short_edge", None
        )

        if max_scale_long_edge is None and max_scale_short_edge is not None:
            logger.warning(
                "max_scale_long_edge cannot be none when max_scale_short_edge is not None in {}. Setting both to "
                "None".format(self.__class__.__name__)
            )
            max_scale_long_edge = None
            max_scale_short_edge = None
        elif max_scale_long_edge is not None and max_scale_short_edge is None:
            logger.warning(
                "max_scale_short_edge cannot be none when max_scale_long_edge is not None in {}. Setting both to "
                "None".format(self.__class__.__name__)
            )
            max_scale_long_edge = None
            max_scale_short_edge = None

        super().__init__(opts=opts)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.max_scale_long_edge = max_scale_long_edge
        self.max_scale_short_edge = max_scale_short_edge

        self.interpolation = interpolation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.random-resize.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-resize.max-scale-long-edge",
            type=int,
            default=None,
            help="Max. value along the longest edge. Defaults to None",
        )
        group.add_argument(
            "--image-augmentation.random-resize.max-scale-short-edge",
            type=int,
            default=None,
            help="Max. value along the shortest edge. Defaults to None.",
        )

        group.add_argument(
            "--image-augmentation.random-resize.min-ratio",
            type=float,
            default=0.5,
            help="Min ratio for random resizing. Defaults to 0.5",
        )
        group.add_argument(
            "--image-augmentation.random-resize.max-ratio",
            type=float,
            default=2.0,
            help="Max ratio for random resizing. Defaults to 2.0",
        )
        group.add_argument(
            "--image-augmentation.random-resize.interpolation",
            type=str,
            default="bilinear",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Desired interpolation method. Defaults to bilinear.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        random_ratio = random.uniform(self.min_ratio, self.max_ratio)

        # compute the size
        width, height = F.get_image_size(data["image"])
        if self.max_scale_long_edge is not None:
            min_hw = min(height, width)
            max_hw = max(height, width)
            scale_factor = (
                min(
                    self.max_scale_long_edge / max_hw,
                    self.max_scale_short_edge / min_hw,
                )
                * random_ratio
            )
            # resize while maintaining aspect ratio
            new_size = int(math.ceil(height * scale_factor)), int(
                math.ceil(width * scale_factor)
            )
        else:
            new_size = int(math.ceil(height * random_ratio)), int(
                math.ceil(width * random_ratio)
            )
        # new_size should be a tuple of height and width
        return _resize_fn(data, size=new_size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(min_ratio={}, max_ratio={}, interpolation={}, max_long_edge={}, max_short_edge={})".format(
            self.__class__.__name__,
            self.min_ratio,
            self.max_ratio,
            self.interpolation,
            self.max_scale_long_edge,
            self.max_scale_short_edge,
        )


@register_transformations(name="random_short_size_resize", type="image_pil")
class RandomShortSizeResize(BaseTransformation):
    """
    This class implements random resizing such that shortest side is between specified minimum and maximum values.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        short_size_min = getattr(
            opts, "image_augmentation.random_short_size_resize.short_side_min", None
        )
        short_size_max = getattr(
            opts, "image_augmentation.random_short_size_resize.short_side_max", None
        )
        max_img_dim = getattr(
            opts, "image_augmentation.random_short_size_resize.max_img_dim", None
        )
        if short_size_min is None:
            logger.error(
                "Short side minimum value can't be None in {}".format(
                    self.__class__.__name__
                )
            )
        if short_size_max is None:
            logger.error(
                "Short side maximum value can't be None in {}".format(
                    self.__class__.__name__
                )
            )
        if max_img_dim is None:
            logger.error(
                "Max. image dimension value can't be None in {}".format(
                    self.__class__.__name__
                )
            )

        if short_size_max <= short_size_min:
            logger.error(
                "Short side maximum value should be >= short side minimum value in {}. Got: {} and {}".format(
                    self.__class__.__name__, short_size_max, short_size_min
                )
            )

        interpolation = getattr(
            opts, "image_augmentation.random_short_size_resize.interpolation", "bicubic"
        )

        self.short_side_min = short_size_min
        self.short_side_max = short_size_max
        self.max_img_dim = max_img_dim
        self.interpolation = interpolation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.random-short-size-resize.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-short-size-resize.short-side-min",
            type=int,
            default=None,
            help="Minimum value for image's shortest side. Defaults to None.",
        )
        group.add_argument(
            "--image-augmentation.random-short-size-resize.short-side-max",
            type=int,
            default=None,
            help="Maximum value for image's shortest side. Defaults to None.",
        )
        group.add_argument(
            "--image-augmentation.random-short-size-resize.interpolation",
            type=str,
            default="bicubic",
            choices=list(INTERPOLATION_MODE_MAP.keys()),
            help="Desired interpolation method. Defaults to bicubic",
        )
        group.add_argument(
            "--image-augmentation.random-short-size-resize.max-img-dim",
            type=int,
            default=None,
            help="Max. image dimension. Defaults to None.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        short_side = random.randint(self.short_side_min, self.short_side_max)
        img_w, img_h = data["image"].size
        scale = min(
            short_side / min(img_h, img_w), self.max_img_dim / max(img_h, img_w)
        )
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)
        data = _resize_fn(data, size=(img_h, img_w), interpolation=self.interpolation)
        return data

    def __repr__(self) -> str:
        return "{}(short_side_min={}, short_side_max={}, interpolation={})".format(
            self.__class__.__name__,
            self.short_side_min,
            self.short_side_max,
            self.interpolation,
        )


@register_transformations(name="random_erasing", type="image_pil")
class RandomErasing(BaseTransformation, T.RandomErasing):
    """
    This class randomly selects a region in a tensor and erases its pixels.
    See `this paper <https://arxiv.org/abs/1708.04896>`_ for details.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        BaseTransformation.__init__(self, opts=opts)
        random_erase_p = getattr(opts, "image_augmentation.random_erase.p", 0.5)
        T.RandomErasing.__init__(self, p=random_erase_p)

        self.random_erase_p = random_erase_p

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.random-erase.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-erase.p",
            type=float,
            default=0.5,
            help="Probability that random erasing operation will be applied. Defaults to 0.5",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        data["image"] = super().forward(data.pop("image"))
        return data

    def __repr__(self) -> str:
        return "{}(random_erase_p={})".format(
            self.__class__.__name__, self.random_erase_p
        )


@register_transformations(name="random_gaussian_blur", type="image_pil")
class RandomGaussianBlur(BaseTransformation):
    """
    This method randomly blurs the input image.
    """

    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts=opts)
        self.p = getattr(opts, "image_augmentation.random_gaussian_noise.p", 0.5)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.random-gaussian-noise.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-gaussian-noise.p",
            type=float,
            default=0.5,
            help="Probability for applying {}".format(cls.__name__),
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        if random.random() < self.p:
            img = data.pop("image")
            # radius is the standard devaition of the gaussian kernel
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            data["image"] = img
        return data


@register_transformations(name="random_crop", type="image_pil")
class RandomCrop(BaseTransformation):
    """
    This method randomly crops an image area.

    .. note::
        If the size of input image is smaller than the desired crop size, the input image is first resized
        while maintaining the aspect ratio and then cropping is performed.
    """

    def __init__(
        self,
        opts,
        size: Union[Sequence, int],
        ignore_idx: Optional[int] = 255,
        *args,
        **kwargs
    ) -> None:
        super().__init__(opts=opts)
        self.height, self.width = setup_size(size=size)
        self.opts = opts
        self.seg_class_max_ratio = getattr(
            opts, "image_augmentation.random_crop.seg_class_max_ratio", None
        )
        self.ignore_idx = ignore_idx
        self.num_repeats = 10
        self.seg_fill = getattr(opts, "image_augmentation.random_crop.mask_fill", 0)
        pad_if_needed = getattr(
            opts, "image_augmentation.random_crop.pad_if_needed", False
        )
        self.if_needed_fn = (
            self._pad_if_needed if pad_if_needed else self._resize_if_needed
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--image-augmentation.random-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-crop.seg-class-max-ratio",
            default=None,
            type=float,
            help="Max. ratio that single segmentation class can occupy. Defaults to None",
        )
        group.add_argument(
            "--image-augmentation.random-crop.pad-if-needed",
            action="store_true",
            help="Pad images if needed. Defaults to False, i.e., resizing will be performed",
        )
        group.add_argument(
            "--image-augmentation.random-crop.mask-fill",
            type=int,
            default=255,
            help="Value to fill in segmentation mask in case of padding. Defaults to 255. "
            "Generally, this value is the same as background or undefined class id.",
        )
        return parser

    @staticmethod
    def get_params(img_h, img_w, target_h, target_w):
        if img_w == target_w and img_h == target_h:
            return 0, 0, img_h, img_w

        i = random.randint(0, max(0, img_h - target_h))
        j = random.randint(0, max(0, img_w - target_w))
        return i, j, target_h, target_w

    @staticmethod
    def get_params_from_box(boxes, img_h, img_w):
        # x, y, w, h
        offset = random.randint(20, 50)
        start_x = max(0, int(round(np.min(boxes[..., 0]))) - offset)
        start_y = max(0, int(round(np.min(boxes[..., 1]))) - offset)
        end_x = min(int(round(np.max(boxes[..., 2]))) + offset, img_w)
        end_y = min(int(round(np.max(boxes[..., 3]))) + offset, img_h)

        return start_y, start_x, end_y - start_y, end_x - start_x

    def get_params_from_mask(self, data, i, j, h, w):
        img_w, img_h = F.get_image_size(data["image"])
        for _ in range(self.num_repeats):
            temp_data = _crop_fn(
                data=copy.deepcopy(data), top=i, left=j, height=h, width=w
            )
            class_labels, cls_count = np.unique(
                np.array(temp_data["mask"]), return_counts=True
            )
            valid_cls_count = cls_count[class_labels != self.ignore_idx]

            if valid_cls_count.size == 0:
                continue

            # compute the ratio of segmentation class with max. pixels to total pixels.
            # If the ratio is less than seg_class_max_ratio, then exit the loop
            total_valid_pixels = np.sum(valid_cls_count)
            max_valid_pixels = np.max(valid_cls_count)
            ratio = max_valid_pixels / total_valid_pixels

            if len(cls_count) > 1 and ratio < self.seg_class_max_ratio:
                break
            i, j, h, w = self.get_params(
                img_h=img_h, img_w=img_w, target_h=self.height, target_w=self.width
            )
        return i, j, h, w

    def _resize_if_needed(self, data: Dict) -> Dict:
        img = data["image"]

        w, h = F.get_image_size(img)
        # resize while maintaining the aspect ratio
        new_size = min(h + max(0, self.height - h), w + max(0, self.width - w))

        return _resize_fn(
            data, size=new_size, interpolation=T.InterpolationMode.BILINEAR
        )

    def _pad_if_needed(self, data: Dict) -> Dict:
        img = data.pop("image")

        w, h = F.get_image_size(img)
        new_h = h + max(self.height - h, 0)
        new_w = w + max(self.width - w, 0)

        pad_img = Image.new(img.mode, (new_w, new_h), color=0)
        pad_img.paste(img, (0, 0))
        data["image"] = pad_img

        if "mask" in data:
            mask = data.pop("mask")
            pad_mask = Image.new(mask.mode, (new_w, new_h), color=self.seg_fill)
            pad_mask.paste(mask, (0, 0))
            data["mask"] = pad_mask

        return data

    def __call__(self, data: Dict) -> Dict:
        # box_info
        if "box_coordinates" in data:
            boxes = data.get("box_coordinates")
            # crop the relevant area
            image_w, image_h = F.get_image_size(data["image"])
            box_i, box_j, box_h, box_w = self.get_params_from_box(
                boxes, image_h, image_w
            )
            data = _crop_fn(data, top=box_i, left=box_j, height=box_h, width=box_w)

        data = self.if_needed_fn(data)

        img_w, img_h = F.get_image_size(data["image"])
        i, j, h, w = self.get_params(
            img_h=img_h, img_w=img_w, target_h=self.height, target_w=self.width
        )

        if (
            "mask" in data
            and self.seg_class_max_ratio is not None
            and self.seg_class_max_ratio < 1.0
        ):
            i, j, h, w = self.get_params_from_mask(data=data, i=i, j=j, h=h, w=w)

        data = _crop_fn(data=data, top=i, left=j, height=h, width=w)
        return data

    def __repr__(self) -> str:
        return "{}(size=(h={}, w={}), seg_class_max_ratio={}, seg_fill={})".format(
            self.__class__.__name__,
            self.height,
            self.width,
            self.seg_class_max_ratio,
            self.seg_fill,
        )


@register_transformations(name="to_tensor", type="image_pil")
class ToTensor(BaseTransformation):
    """
    This method converts an image into a tensor.

    .. note::
        We do not perform any mean-std normalization. If mean-std normalization is desired, please modify this class.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        img_dtype = getattr(opts, "image_augmentation.to_tensor.dtype", "float")
        self.img_dtype = torch.float
        if img_dtype in ["half", "float16"]:
            self.img_dtype = torch.float16

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--image-augmentation.to-tensor.dtype",
            type=str,
            default="float",
            help="Tensor data type. Default is float",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        # HWC --> CHW
        img = data["image"]

        if F._is_pil_image(img):
            # convert PIL image to tensor
            img = F.pil_to_tensor(img).contiguous()

        data["image"] = img.to(dtype=self.img_dtype).div(255.0)

        if "mask" in data:
            mask = data.pop("mask")
            mask = np.array(mask)
            if len(mask.shape) > 2 and mask.shape[-1] > 1:
                mask = np.ascontiguousarray(mask.transpose(2, 0, 1))
            data["mask"] = torch.as_tensor(mask, dtype=torch.long)

        if "box_coordinates" in data:
            boxes = data.pop("box_coordinates")
            data["box_coordinates"] = torch.as_tensor(boxes, dtype=torch.float)

        if "box_labels" in data:
            box_labels = data.pop("box_labels")
            data["box_labels"] = torch.as_tensor(box_labels)

        if "instance_mask" in data:
            assert "instance_coords" in data
            instance_masks = data.pop("instance_mask")
            data["instance_mask"] = instance_masks.to(dtype=torch.long)

            instance_coords = data.pop("instance_coords")
            data["instance_coords"] = torch.as_tensor(
                instance_coords, dtype=torch.float
            )
        return data


@register_transformations(name="compose", type="image_pil")
class Compose(BaseTransformation):
    """
    This method applies a list of transforms in a sequential fashion.
    """

    def __init__(self, opts, img_transforms: List, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.img_transforms = img_transforms

    def __call__(self, data: Dict) -> Dict:
        for t in self.img_transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        transform_str = ", ".join("\n\t\t\t" + str(t) for t in self.img_transforms)
        repr_str = "{}({}\n\t\t)".format(self.__class__.__name__, transform_str)
        return repr_str


@register_transformations(name="random_order", type="image_pil")
class RandomOrder(BaseTransformation):
    """
    This method applies a list of all or few transforms in a random order.
    """

    def __init__(self, opts, img_transforms: List, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.transforms = img_transforms
        apply_k_factor = getattr(opts, "image_augmentation.random_order.apply_k", 1.0)
        assert (
            0.0 < apply_k_factor <= 1.0
        ), "--image-augmentation.random-order.apply-k should be > 0 and <= 1"
        self.keep_t = int(math.ceil(len(self.transforms) * apply_k_factor))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--image-augmentation.random-order.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-order.apply-k",
            type=int,
            default=1.0,
            help="Apply K percent of transforms randomly. Value between 0 and 1. "
            "Defaults to 1 (i.e., apply all transforms in random order).",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        random.shuffle(self.transforms)
        for t in self.transforms[: self.keep_t]:
            data = t(data)
        return data

    def __repr__(self):
        transform_str = ", ".join(str(t) for t in self.transforms)
        repr_str = "{}(n_transforms={}, t_list=[{}]".format(
            self.__class__.__name__, self.keep_t, transform_str
        )
        return repr_str

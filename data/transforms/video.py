#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import torch
import math
import argparse
from typing import Sequence, Dict, Any, Union, Tuple, List, Optional
from torch.nn import functional as F

from utils import logger

from . import register_transformations, BaseTransformation
from .utils import *


SUPPORTED_PYTORCH_INTERPOLATIONS = ["nearest", "bilinear", "bicubic"]


def _check_interpolation(interpolation):
    if interpolation not in SUPPORTED_PYTORCH_INTERPOLATIONS:
        inter_str = "Supported interpolation modes are:"
        for i, j in enumerate(SUPPORTED_PYTORCH_INTERPOLATIONS):
            inter_str += "\n\t{}: {}".format(i, j)
        logger.error(inter_str)
    return interpolation


def _crop_fn(data: Dict, i: int, j: int, h: int, w: int):
    img = data["image"]
    if not isinstance(img, torch.Tensor) and img.dim() != 4:
        logger.error(
            "Cropping requires 4-d tensor of shape NCHW or CNHW. Got {}-dimensional tensor".format(
                img.dim()
            )
        )

    crop_image = img[..., i : i + h, j : j + w]
    data["image"] = crop_image

    mask = data.get("mask", None)
    if mask is not None:
        crop_mask = mask[..., i : i + h, j : j + w]
        data["mask"] = crop_mask
    return data


def _resize_fn(
    data: Dict, size: Union[Sequence, int], interpolation: Optional[str] = "bilinear"
):
    img = data["image"]

    if isinstance(size, Sequence) and len(size) == 2:
        size_h, size_w = size[0], size[1]
    elif isinstance(size, int):
        h, w = img.shape[-2:]
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
        interpolation = _check_interpolation(interpolation)
    img = F.interpolate(
        input=img,
        size=(size_w, size_h),
        mode=interpolation,
        align_corners=True if interpolation != "nearest" else None,
    )
    data["image"] = img

    mask = data.get("mask", None)
    if mask is not None:
        mask = F.interpolate(input=mask, size=(size_w, size_h), mode="nearest")
        data["mask"] = mask

    return data


def _check_rgb_video_tensor(clip):
    if not isinstance(clip, torch.FloatTensor) or clip.dim() != 4:
        logger.error(
            "Video clip is either not an instance of FloatTensor or it is not a 4-d tensor (NCHW or CNHW)"
        )


@register_transformations(name="to_tensor", type="video")
class ToTensor(BaseTransformation):
    """
    This method converts an image into a tensor.

    .. note::
        We do not perform any mean-std normalization. If mean-std normalization is desired, please modify this class.
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)

    def __call__(self, data: Dict) -> Dict:
        # [C, N, H, W]
        clip = data["image"]
        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip)
        clip = clip.float()

        _check_rgb_video_tensor(clip=clip)

        # normalize between 0 and 1
        clip = torch.div(clip, 255.0)
        data["image"] = clip
        return data


@register_transformations(name="random_resized_crop", type="video")
class RandomResizedCrop(BaseTransformation):
    """
    This class crops a random portion of an image and resize it to a given size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        interpolation = getattr(
            opts, "video_augmentation.random_resized_crop.interpolation", "bilinear"
        )
        scale = getattr(
            opts, "video_augmentation.random_resized_crop.scale", (0.08, 1.0)
        )
        ratio = getattr(
            opts,
            "video_augmentation.random_resized_crop.aspect_ratio",
            (3.0 / 4.0, 4.0 / 3.0),
        )

        if not isinstance(scale, Sequence) or (
            isinstance(scale, Sequence)
            and len(scale) != 2
            and 0.0 <= scale[0] < scale[1]
        ):
            logger.error(
                "--video-augmentation.random-resized-crop.scale should be a tuple of length 2 "
                "such that 0.0 <= scale[0] < scale[1]. Got: {}".format(scale)
            )

        if not isinstance(ratio, Sequence) or (
            isinstance(ratio, Sequence)
            and len(ratio) != 2
            and 0.0 < ratio[0] < ratio[1]
        ):
            logger.error(
                "--video-augmentation.random-resized-crop.aspect-ratio should be a tuple of length 2 "
                "such that 0.0 < ratio[0] < ratio[1]. Got: {}".format(ratio)
            )

        ratio = (round(ratio[0], 3), round(ratio[1], 3))

        super().__init__(opts=opts)

        self.scale = scale
        self.size = setup_size(size=size)

        self.interpolation = _check_interpolation(interpolation)
        self.ratio = ratio

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--video-augmentation.random-resized-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear",
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.scale",
            type=tuple,
            default=(0.08, 1.0),
            help="Specifies the lower and upper bounds for the random area of the crop, before resizing."
            " The scale is defined with respect to the area of the original image. Defaults to "
            "(0.08, 1.0)",
        )
        group.add_argument(
            "--video-augmentation.random-resized-crop.aspect-ratio",
            type=float or tuple,
            default=(3.0 / 4.0, 4.0 / 3.0),
            help="lower and upper bounds for the random aspect ratio of the crop, before resizing. "
            "Defaults to (3./4., 4./3.)",
        )
        return parser

    def get_params(self, height: int, width: int) -> (int, int, int, int):
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = (1.0 * width) / height
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data: Dict) -> Dict:
        clip = data["image"]
        _check_rgb_video_tensor(clip=clip)

        height, width = clip.shape[-2:]

        i, j, h, w = self.get_params(height=height, width=width)
        data = _crop_fn(data=data, i=i, j=j, h=h, w=w)
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return "{}(scale={}, ratio={}, interpolation={})".format(
            self.__class__.__name__, self.scale, self.ratio, self.interpolation
        )


@register_transformations(name="random_short_side_resize_crop", type="video")
class RandomShortSizeResizeCrop(BaseTransformation):
    """
    This class first randomly resizes the input video such that shortest side is between specified minimum and
    maximum values, adn then crops a desired size video.

    .. note::
        This class assumes that the video size after resizing is greater than or equal to the desired size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        interpolation = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.interpolation",
            "bilinear",
        )
        short_size_min = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.short_side_min",
            None,
        )
        short_size_max = getattr(
            opts,
            "video_augmentation.random_short_side_resize_crop.short_side_max",
            None,
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

        if short_size_max <= short_size_min:
            logger.error(
                "Short side maximum value should be >= short side minimum value in {}. Got: {} and {}".format(
                    self.__class__.__name__, short_size_max, short_size_min
                )
            )

        super().__init__(opts=opts)
        self.short_side_min = short_size_min
        self.size = size
        self.short_side_max = short_size_max
        self.interpolation = _check_interpolation(interpolation)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Desired interpolation method. Defaults to bilinear",
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.short-side-min",
            type=int,
            default=None,
            help="Minimum value for video's shortest side. Defaults to None.",
        )
        group.add_argument(
            "--video-augmentation.random-short-side-resize-crop.short-side-max",
            type=int,
            default=None,
            help="Maximum value for video's shortest side. Defaults to None.",
        )
        return parser

    def get_params(self, height, width) -> Tuple[int, int, int, int]:
        th, tw = self.size

        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def __call__(self, data: Dict) -> Dict:
        short_dim = random.randint(self.short_side_max, self.short_side_max)
        # resize the video so that shorter side is short_dim
        data = _resize_fn(data, size=short_dim, interpolation=self.interpolation)

        clip = data["image"]
        _check_rgb_video_tensor(clip=clip)
        height, width = clip.shape[-2:]
        i, j, h, w = self.get_params(height=height, width=width)
        # crop the video
        return _crop_fn(data=data, i=i, j=j, h=h, w=w)

    def __repr__(self) -> str:
        return "{}(size={}, short_size_range=({}, {}), interpolation={})".format(
            self.__class__.__name__,
            self.size,
            self.short_side_min,
            self.short_side_max,
            self.interpolation,
        )


@register_transformations(name="random_crop", type="video")
class RandomCrop(BaseTransformation):
    """
    This method randomly crops a video area.

    .. note::
        This class assumes that the input video size is greater than or equal to the desired size.
    """

    def __init__(self, opts, size: Union[Tuple, int], *args, **kwargs) -> None:
        size = setup_size(size=size)
        super().__init__(opts=opts)
        self.size = size

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--video-augmentation.random-crop.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        return parser

    def get_params(self, height, width) -> Tuple[int, int, int, int]:
        th, tw = self.size

        if width == tw and height == th:
            return 0, 0, height, width

        i = random.randint(0, height - th)
        j = random.randint(0, width - tw)
        return i, j, th, tw

    def __call__(self, data: Dict) -> Dict:
        clip = data["image"]
        _check_rgb_video_tensor(clip=clip)
        height, width = clip.shape[-2:]
        i, j, h, w = self.get_params(height=height, width=width)
        return _crop_fn(data=data, i=i, j=j, h=h, w=w)

    def __repr__(self) -> str:
        return "{}(size={})".format(self.__class__.__name__, self.size)


@register_transformations(name="random_horizontal_flip", type="video")
class RandomHorizontalFlip(BaseTransformation):
    """
    This class implements random horizontal flipping method
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        p = getattr(opts, "video_augmentation.random_horizontal_flip.p", 0.5)
        super().__init__(opts=opts)
        self.p = p

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--video-augmentation.random-horizontal-flip.enable",
            action="store_true",
            help="use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--video-augmentation.random-horizontal-flip.p",
            type=float,
            default=0.5,
            help="Probability for random horizontal flip",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:

        if random.random() <= self.p:
            clip = data["image"]
            _check_rgb_video_tensor(clip=clip)
            clip = torch.flip(clip, dims=[-1])
            data["image"] = clip

            mask = data.get("mask", None)
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
                data["mask"] = mask

        return data


@register_transformations(name="center_crop", type="video")
class CenterCrop(BaseTransformation):
    """
    This class implements center cropping method.

    .. note::
        This class assumes that the input size is greater than or equal to the desired size.
    """

    def __init__(self, opts, size: Sequence or int, *args, **kwargs) -> None:
        super().__init__(opts=opts)
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
            "--video-augmentation.center-crop.enable",
            action="store_true",
            help="use center cropping",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        height, width = data["image"].shape[-2:]
        i = (height - self.height) // 2
        j = (width - self.width) // 2
        return _crop_fn(data=data, i=i, j=j, h=self.height, w=self.width)

    def __repr__(self) -> str:
        return "{}(size=(h={}, w={}))".format(
            self.__class__.__name__, self.height, self.width
        )


@register_transformations(name="resize", type="video")
class Resize(BaseTransformation):
    """
    This class implements resizing operation.

    .. note::
    Two possible modes for resizing.
    1. Resize while maintaining aspect ratio. To enable this option, pass int as a size
    2. Resize to a fixed size. To enable this option, pass a tuple of height and width as a size
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        size = getattr(opts, "video_augmentation.resize.size", None)
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

        interpolation = getattr(
            opts, "video_augmentation.resize.interpolation", "bilinear"
        )
        super().__init__(opts=opts)

        self.size = size
        self.interpolation = _check_interpolation(interpolation)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--video-augmentation.resize.enable",
            action="store_true",
            help="use fixed resizing",
        )

        group.add_argument(
            "--video-augmentation.resize.interpolation",
            type=str,
            default="bilinear",
            choices=SUPPORTED_PYTORCH_INTERPOLATIONS,
            help="Interpolation for resizing. Default is bilinear",
        )
        group.add_argument(
            "--video-augmentation.resize.size",
            type=int,
            nargs="+",
            default=None,
            help="Resize video to the specified size. If int is passed, then shorter side is resized"
            "to the specified size and longest side is resized while maintaining aspect ratio."
            "Defaults to None.",
        )
        return parser

    def __call__(self, data: Dict) -> Dict:
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self):
        return "{}(size={}, interpolation={})".format(
            self.__class__.__name__, self.size, self.interpolation
        )


@register_transformations(name="compose", type="video")
class Compose(BaseTransformation):
    """
    This method applies a list of transforms in a sequential fashion.
    """

    def __init__(self, opts, video_transforms: List, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.video_transforms = video_transforms

    def __call__(self, data: Dict) -> Dict:
        for t in self.video_transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        transform_str = ", ".join("\n\t\t\t" + str(t) for t in self.video_transforms)
        repr_str = "{}({})".format(self.__class__.__name__, transform_str)
        return repr_str

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import cv2
from torch.utils import data
from typing import Optional
import argparse

from ..transforms import image as cv_transforms


class BaseImageDataset(data.Dataset):
    """
        Base Dataset class for Image datasets
    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False, *args,
                 **kwargs):
        root = getattr(opts, "dataset.root_train", None) if is_training else getattr(opts, "dataset.root_val", None)
        self.root = root
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.sampler_name = getattr(opts, "sampler.name", None)
        self.opts = opts

    @staticmethod
    def additional_transforms(opts):
        aug_list = []

        if getattr(opts, "image_augmentation.random_gamma_correction.enable", False):
            aug_list.append(cv_transforms.RandomGammaCorrection(opts=opts))

        if getattr(opts, "image_augmentation.random_rotate.enable", False):
            aug_list.append(cv_transforms.RandomRotate(opts=opts))

        if getattr(opts, "image_augmentation.random_blur.enable", False):
            aug_list.append(cv_transforms.RandomBlur(opts=opts))

        if getattr(opts, "image_augmentation.random_translate.enable", False):
            aug_list.append(cv_transforms.RandomTranslate(opts=opts))

        if getattr(opts, "image_augmentation.random_jpeg_compress.enable", False):
            aug_list.append(cv_transforms.RandomJPEGCompress(opts=opts))

        if getattr(opts, "image_augmentation.random_gauss_noise.enable", False):
            aug_list.append(cv_transforms.RandomGaussianNoise(opts=opts))

        if getattr(opts, "image_augmentation.random_resize.enable", False):
            aug_list.append(cv_transforms.RandomResize(opts=opts))

        if getattr(opts, "image_augmentation.random_scale.enable", False):
            aug_list.append(cv_transforms.RandomScale(opts=opts))

        if getattr(opts, "image_augmentation.photo_metric_distort.enable", False):
            aug_list.append(cv_transforms.PhotometricDistort(opts=opts))

        # Flipping
        random_flip = getattr(opts, "image_augmentation.random_flip.enable", False)
        if random_flip:
            aug_list.append(cv_transforms.RandomFlip(opts=opts))
        else:
            random_h_flip = getattr(opts, "image_augmentation.random_horizontal_flip.enable", False)
            random_v_flip = getattr(opts, "image_augmentation.random_vertical_flip.enable", False)

            if random_h_flip and random_v_flip:
                aug_list.append(cv_transforms.RandomFlip(opts=opts))
            elif random_v_flip:
                aug_list.append(cv_transforms.RandomVerticalFlip(opts=opts))
            elif random_h_flip:
                aug_list.append(cv_transforms.RandomHorizontalFlip(opts=opts))

        if getattr(opts, "image_augmentation.random_order.enable", False):
            assert len(aug_list) > 1
            aug_list = [cv_transforms.RandomOrder(opts=opts, img_transforms=aug_list)]

        return aug_list

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def training_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def validation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def evaluation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def read_image(path: str):
        return cv2.imread(path, cv2.IMREAD_COLOR) # Image is read in BGR Format and not RGB format

    @staticmethod
    def read_mask(path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def __repr__(self):
        return "{}(root={}, is_training={})".format(self.__class__.__name__, self.root, self.is_training)

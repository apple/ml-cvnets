#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from torchvision.datasets import ImageFolder
from typing import Optional, Tuple, Dict
import torch
import numpy as np

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image as tf


@register_dataset(name="imagenet", task="classification")
class ImagenetDataset(BaseImageDataset, ImageFolder):
    """
        Dataset class for the ImageNet dataset.

        Dataset structure

        + imagenet
          |- training
             |- n*
          |- validation
             |- n*
        Both validation and training will have 1000 folders starting with 'n' (1 folder per class).
    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False) -> None:
        """

        :param opts: arguments
        :param is_training: Training or validation mode
        :param is_evaluation: Evaluation mode
        """
        BaseImageDataset.__init__(self, opts=opts, is_training=is_training, is_evaluation=is_evaluation)
        root = self.root
        ImageFolder.__init__(self, root=root, transform=None, target_transform=None, is_valid_file=None)

    def training_transforms(self, size: tuple or int):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        aug_list = [tf.RandomResizedCrop(opts=self.opts, size=size)]
        aug_list.extend(self.additional_transforms(opts=self.opts))
        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size: tuple):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        if isinstance(size, (tuple, list)):
            size = min(size)

        assert isinstance(size, int)
        # (256 - 224) = 32
        # where 224/0.875 = 256
        scale_size = size + 32 # int(make_divisible(crop_size / 0.875, divisor=32))

        return tf.Compose(opts=self.opts, img_transforms=[
            tf.Resize(opts=self.opts, size=scale_size),
            tf.CenterCrop(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ])

    def evaluation_transforms(self, size: tuple):
        """

        :param size: crop size (H, W)
        :return: list of augmentation methods
        """
        return self.validation_transforms(size=size)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """

        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image and label ID.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[img_index]
        input_img = self.read_image(img_path)

        if input_img is None:
            # Sometimes images are corrupt and cv2 is not able to load them
            # Skip such images
            logger.log('Img index {} is possibly corrupt. Removing it from the sample list'.format(img_index))
            del self.samples[img_index]
            input_img = np.zeros(shape=(crop_size_h, crop_size_w, 3), dtype=np.uint8)

        data = {"image": input_img}
        data = transform_fn(data)

        # target is a 0-dimensional tensor
        target_tensor = torch.tensor(1, dtype=torch.long).fill_(target)

        data["label"] = target_tensor
        return data

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        from utils.tensor_utils import tensor_size_from_opts
        im_h, im_w = tensor_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self.training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self.validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\t is_training={}\n\tsamples={}\n\ttransforms={}\n)".format(self.__class__.__name__,
                                                                                            self.root,
                                                                                            self.is_training,
                                                                                            len(self.samples),
                                                                                            transforms_str)

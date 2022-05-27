#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torchvision.datasets import ImageFolder
from typing import Optional, Tuple, Dict
import numpy as np
import math

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_opencv as tf


@register_dataset(name="imagenet_opencv", task="classification")
class ImagenetOpenCVDataset(BaseImageDataset, ImageFolder):
    """
    ImageNet Classification Dataset that uses OpenCV for data augmentation.

    The dataset structure should follow the ImageFolder class in :class:`torchvision.datasets.imagenet`

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    .. note::
        This class is depreciated and will be removed in future versions (Use it for MobileViT evaluation).
    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        BaseImageDataset.__init__(
            self, opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        root = self.root
        ImageFolder.__init__(
            self, root=root, transform=None, target_transform=None, is_valid_file=None
        )

        self.n_classes = len(list(self.class_to_idx.keys()))
        setattr(opts, "model.classification.n_classes", self.n_classes)

    def _training_transforms(self, size: tuple or int):
        """
        Training data augmentation methods (RandomResizedCrop --> RandomHorizontalFlip --> ToTensor).
        """
        aug_list = [
            tf.RandomResizedCrop(opts=self.opts, size=size),
            tf.RandomHorizontalFlip(opts=self.opts),
            tf.NumpyToTensor(opts=self.opts),
        ]
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple):
        """Implements validation transformation method (Resize --> CenterCrop --> ToTensor)."""
        if isinstance(size, (tuple, list)):
            size = min(size)

        assert isinstance(size, int)
        # (256 - 224) = 32
        # where 224/0.875 = 256

        crop_ratio = getattr(self.opts, "dataset.imagenet.crop_ratio", 0.875)
        if 0 < crop_ratio < 1.0:
            scale_size = int(math.ceil(size / crop_ratio))
            scale_size = (scale_size // 32) * 32
        else:
            logger.warning(
                "Crop ratio should be between 0 and 1. Got: {}".format(crop_ratio)
            )
            logger.warning("Setting scale_size as size + 32")
            scale_size = size + 32  # int(make_divisible(crop_size / 0.875, divisor=32))

        return tf.Compose(
            opts=self.opts,
            img_transforms=[
                tf.Resize(opts=self.opts, size=scale_size),
                tf.CenterCrop(opts=self.opts, size=size),
                tf.NumpyToTensor(opts=self.opts),
            ],
        )

    def _evaluation_transforms(self, size: tuple):
        """Same as the validation_transforms"""
        return self._validation_transforms(size=size)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """

        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image and label ID.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:  # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[img_index]
        input_img = self.read_image_opencv(img_path)

        if input_img is None:
            # Sometimes images are corrupt and cv2 is not able to load them
            # Skip such images
            logger.log(
                "Img index {} is possibly corrupt. Removing it from the sample list".format(
                    img_index
                )
            )
            del self.samples[img_index]
            input_img = np.zeros(shape=(crop_size_h, crop_size_w, 3), dtype=np.uint8)

        data = {"image": input_img}
        data = transform_fn(data)

        data["label"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tn_classes={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.samples),
            self.n_classes,
            transforms_str,
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torchvision.datasets import ImageFolder
from typing import Optional, Tuple, Dict, List, Union
import torch
import argparse

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T
from ...collate_fns import register_collate_fn


@register_dataset(name="imagenet", task="classification")
class ImagenetDataset(BaseImageDataset, ImageFolder):
    """
    ImageNet Classification Dataset that uses PIL for reading and augmenting images. The dataset structure should
    follow the ImageFolder class in :class:`torchvision.datasets.imagenet`

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    .. note::
        We recommend to use this dataset class over the imagenet_opencv.py file.

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
        setattr(opts, "dataset.collate_fn_name_train", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "imagenet_collate_fn")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.imagenet.crop-ratio",
            type=float,
            default=0.875,
            help="Crop ratio",
        )
        return parser

    def _training_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
            Training data augmentation methods.
                Image --> RandomResizedCrop --> RandomHorizontalFlip --> Optional(AutoAugment or RandAugment)
                --> Tensor --> Optional(RandomErasing) --> Optional(MixUp) --> Optional(CutMix)

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
            T.RandomHorizontalFlip(opts=self.opts),
        ]
        auto_augment = getattr(
            self.opts, "image_augmentation.auto_augment.enable", False
        )
        rand_augment = getattr(
            self.opts, "image_augmentation.rand_augment.enable", False
        )
        if auto_augment and rand_augment:
            logger.error(
                "AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"
            )
        elif auto_augment:
            aug_list.append(T.AutoAugment(opts=self.opts))
        elif rand_augment:
            aug_list.append(T.RandAugment(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable", False):
            aug_list.append(T.RandomErasing(opts=self.opts))

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Validation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """Same as the validation_transforms"""
        return self._validation_transforms(size=size)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[img_index]

        input_img = self.read_image_pil(img_path)

        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=self.img_dtype
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["label"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
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


@register_collate_fn(name="imagenet_collate_fn")
def imagenet_collate_fn(batch: List, opts) -> Dict:
    batch_size = len(batch)
    img_size = [batch_size, *batch[0]["image"].shape]
    img_dtype = batch[0]["image"].dtype

    images = torch.zeros(size=img_size, dtype=img_dtype)
    # fill with -1, so that we can ignore corrupted images
    labels = torch.full(size=[batch_size], fill_value=-1, dtype=torch.long)
    sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
    valid_indexes = []
    for i, batch_i in enumerate(batch):
        label_i = batch_i.pop("label")
        images[i] = batch_i.pop("image")
        labels[i] = label_i  # label is an int
        sample_ids[i] = batch_i.pop("sample_id")  # sample id is an int
        if label_i != -1:
            valid_indexes.append(i)

    valid_indexes = torch.tensor(valid_indexes, dtype=torch.long)
    images = torch.index_select(images, dim=0, index=valid_indexes)
    labels = torch.index_select(labels, dim=0, index=valid_indexes)
    sample_ids = torch.index_select(sample_ids, dim=0, index=valid_indexes)

    channels_last = getattr(opts, "common.channels_last", False)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {
        "image": images,
        "label": labels,
        "sample_id": sample_ids,
        "on_gpu": images.is_cuda,
    }

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
from typing import Optional, List, Tuple, Dict
import argparse
import numpy as np

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T


@register_dataset("pascal", "segmentation")
class PascalVOCDataset(BaseImageDataset):
    """
    Dataset class for the PASCAL VOC 2012 dataset

    The structure of PASCAL VOC dataset should be something like this: ::

        pascal_voc/VOCdevkit/VOC2012/Annotations
        pascal_voc/VOCdevkit/VOC2012/JPEGImages
        pascal_voc/VOCdevkit/VOC2012/SegmentationClass
        pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug_Visualization
        pascal_voc/VOCdevkit/VOC2012/ImageSets
        pascal_voc/VOCdevkit/VOC2012/list
        pascal_voc/VOCdevkit/VOC2012/SegmentationClassAug
        pascal_voc/VOCdevkit/VOC2012/SegmentationObject

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False
    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        use_coco_data = getattr(opts, "dataset.pascal.use_coco_data", False)
        coco_root_dir = getattr(opts, "dataset.pascal.coco_root_dir", None)
        root = self.root

        voc_root_dir = os.path.join(root, "VOC2012")
        voc_list_dir = os.path.join(voc_root_dir, "list")

        coco_data_file = None
        if self.is_training:
            # use the PASCAL VOC 2012 train data with augmented data
            data_file = os.path.join(voc_list_dir, "train_aug.txt")
            if use_coco_data and coco_root_dir is not None:
                coco_data_file = os.path.join(coco_root_dir, "train_2017.txt")
                assert os.path.isfile(
                    coco_data_file
                ), "COCO data file does not exist at: {}".format(coco_root_dir)
        else:
            data_file = os.path.join(voc_list_dir, "val.txt")

        self.images = []
        self.masks = []
        with open(data_file, "r") as lines:
            for line in lines:
                line_split = line.split(" ")
                rgb_img_loc = voc_root_dir + os.sep + line_split[0].strip()
                mask_img_loc = voc_root_dir + os.sep + line_split[1].strip()
                assert os.path.isfile(
                    rgb_img_loc
                ), "RGB file does not exist at: {}".format(rgb_img_loc)
                assert os.path.isfile(
                    mask_img_loc
                ), "Mask image does not exist at: {}".format(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(mask_img_loc)

        # if you want to use Coarse data for training
        if self.is_training and coco_data_file is not None:
            with open(coco_data_file, "r") as lines:
                for line in lines:
                    line_split = line.split(" ")
                    rgb_img_loc = coco_root_dir + os.sep + line_split[0].rstrip()
                    mask_img_loc = coco_root_dir + os.sep + line_split[1].rstrip()
                    assert os.path.isfile(rgb_img_loc)
                    assert os.path.isfile(mask_img_loc)
                    self.images.append(rgb_img_loc)
                    self.masks.append(mask_img_loc)
        self.use_coco_data = use_coco_data
        self.ignore_label = 255
        self.bgrnd_idx = 0
        setattr(opts, "model.segmentation.n_classes", len(self.class_names()))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.pascal.use-coco-data",
            action="store_true",
            help="Use MS-COCO data for training",
        )
        group.add_argument(
            "--dataset.pascal.coco-root-dir",
            type=str,
            default=None,
            help="Location of MS-COCO data",
        )
        return parser

    @staticmethod
    def color_palette():
        color_codes = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]

        color_codes = np.asarray(color_codes).flatten()
        return list(color_codes)

    @staticmethod
    def class_names() -> List:
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted_plant",
            "sheep",
            "sofa",
            "train",
            "tv_monitor",
        ]

    def _training_transforms(self, size: tuple):
        first_aug = T.RandomShortSizeResize(opts=self.opts)
        aug_list = [
            T.RandomHorizontalFlip(opts=self.opts),
            T.RandomCrop(opts=self.opts, size=size, ignore_idx=self.ignore_label),
        ]

        if getattr(self.opts, "image_augmentation.random_gaussian_noise.enable", False):
            aug_list.append(T.RandomGaussianBlur(opts=self.opts))

        if getattr(self.opts, "image_augmentation.photo_metric_distort.enable", False):
            aug_list.append(T.PhotometricDistort(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_rotate.enable", False):
            aug_list.append(T.RandomRotate(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_order.enable", False):
            new_aug_list = [
                first_aug,
                T.RandomOrder(opts=self.opts, img_transforms=aug_list),
                T.ToTensor(opts=self.opts),
            ]
            return T.Compose(opts=self.opts, img_transforms=new_aug_list)
        else:
            aug_list.insert(0, first_aug)
            aug_list.append(T.ToTensor(opts=self.opts))
            return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [T.Resize(opts=self.opts), T.ToTensor(opts=self.opts)]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.segmentation.resize_input_images", False):
            # we want to resize while maintaining aspect ratio. So, we pass img_size argument to resize function
            aug_list.append(T.Resize(opts=self.opts, img_size=min(size)))

        aug_list.append(T.ToTensor(opts=self.opts))
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_training:
            _transform = self._training_transforms(size=crop_size)
        elif self.is_evaluation:
            _transform = self._evaluation_transforms(size=crop_size)
        else:
            _transform = self._validation_transforms(size=crop_size)

        img = self.read_image_pil(self.images[img_index])
        mask = self.read_mask_pil(self.masks[img_index])

        data = {"image": img}
        if not self.is_evaluation:
            data["mask"] = mask

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = self.convert_mask_to_tensor(mask)

        data["label"] = data["mask"]
        del data["mask"]

        if self.is_evaluation:
            im_width, im_height = img.size
            img_name = self.images[img_index].split(os.sep)[-1].replace("jpg", "png")
            data["file_name"] = img_name
            data["im_width"] = im_width
            data["im_height"] = im_height

        return data

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self._evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tuse_coco={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.images),
            self.use_coco_data,
            transforms_str,
        )

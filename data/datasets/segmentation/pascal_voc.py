#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
from typing import Optional
import argparse

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image as tf

VOC_CLASS_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted_plant', 'sheep', 'sofa', 'train',
                  'tv_monitor']


@register_dataset("pascal", "segmentation")
class PascalVOCDataset(BaseImageDataset):
    """
        Dataset class for the PASCAL VOC 2012 dataset

        The structure of PASCAL VOC dataset should be something like this
        + pascal_voc/VOCdevkit/VOC2012/
        + --- Annotations
        + --- JPEGImages
        + --- SegmentationClass
        + --- SegmentationClassAug_Visualization/
        + --- ImageSets
        + --- list
        + --- SegmentationClassAug
        + --- SegmentationObject

    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        """

        :param opts: arguments
        :param is_training: Training or validation mode
        :param is_evaluation: Evaluation mode
        """
        super(PascalVOCDataset, self).__init__(opts=opts, is_training=is_training, is_evaluation=is_evaluation)
        use_coco_data = getattr(opts, "dataset.pascal.use_coco_data", False)
        coco_root_dir = getattr(opts, "dataset.pascal.coco_root_dir", None)
        root = self.root

        voc_root_dir = os.path.join(root, 'VOC2012')
        voc_list_dir = os.path.join(voc_root_dir, 'list')

        coco_data_file = None
        if self.is_training:
            # use the PASCAL VOC 2012 train data with augmented data
            data_file = os.path.join(voc_list_dir, 'train_aug.txt')
            if use_coco_data and coco_root_dir is not None:
                coco_data_file = os.path.join(coco_root_dir, 'train_2017.txt')
                assert os.path.isfile(coco_data_file), 'COCO data file does not exist at: {}'.format(coco_root_dir)
        else:
            data_file = os.path.join(voc_list_dir, 'val.txt')

        self.images = []
        self.masks = []
        with open(data_file, 'r') as lines:
            for line in lines:
                line_split = line.split(" ")
                rgb_img_loc = voc_root_dir + os.sep + line_split[0].strip()
                mask_img_loc = voc_root_dir + os.sep + line_split[1].strip()
                assert os.path.isfile(rgb_img_loc), 'RGB file does not exist at: {}'.format(rgb_img_loc)
                assert os.path.isfile(mask_img_loc), 'Mask image does not exist at: {}'.format(rgb_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(mask_img_loc)

        # if you want to use Coarse data for training
        if self.is_training and coco_data_file is not None:
            with open(coco_data_file, 'r') as lines:
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
        setattr(opts, "model.segmentation.n_classes", len(VOC_CLASS_LIST))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--dataset.pascal.use-coco-data', action='store_true', help='Use MS-COCO data for training')
        group.add_argument('--dataset.pascal.coco-root-dir', type=str, default=None, help='Location of MS-COCO data')
        return parser

    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        aug_list = [
                tf.RandomResize(opts=self.opts),
                tf.RandomCrop(opts=self.opts, size=size),
                tf.RandomHorizontalFlip(opts=self.opts),
                tf.NumpyToTensor(opts=self.opts)
            ]

        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.segmentation.resize_input_images", False):
            aug_list.append(tf.Resize(opts=self.opts, size=size))

        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup):
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_training:
            _transform = self.training_transforms(size=crop_size, ignore_idx=self.ignore_label)
        elif self.is_evaluation:
            _transform = self.evaluation_transforms(size=crop_size)
        else:
            _transform = self.validation_transforms(size=crop_size)

        mask = self.read_mask(self.masks[img_index])
        img = self.read_image(self.images[img_index])

        im_height, im_width = img.shape[:2]

        data = {
            "image": img,
            "mask": None if self.is_evaluation else mask
        }

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        data["label"] = data["mask"]
        del data["mask"]

        if self.is_evaluation:
            img_name = self.images[img_index].split(os.sep)[-1].replace('jpg', 'png')
            data["file_name"] = img_name
            data["im_width"] = im_width
            data["im_height"] = im_height

        return data

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        from utils.tensor_utils import tensor_size_from_opts
        im_h, im_w = tensor_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self.training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self.evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self.validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tuse_coco={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.images),
            self.use_coco_data,
            transforms_str
        )

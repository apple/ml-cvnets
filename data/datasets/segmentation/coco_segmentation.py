#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
from typing import Optional, List, Dict, Union
import argparse

from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import os
from typing import Optional


from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T


@register_dataset("coco", "segmentation")
class COCODataset(BaseImageDataset):
    """
    Dataset class for the COCO dataset that maps classes to PASCAL VOC classes

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
        """

        :param opts: arguments
        :param is_training: Training or validation mode
        :param is_evaluation: Evaluation mode
        """
        super().__init__(
            opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        year = 2017
        split = "train" if is_training else "val"
        ann_file = os.path.join(
            self.root, "annotations/instances_{}{}.json".format(split, year)
        )
        self.img_dir = os.path.join(self.root, "images/{}{}".format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.ids = list(self.coco.imgs.keys())

        self.ignore_label = 255
        self.bgrnd_idx = 0

        setattr(opts, "model.segmentation.n_classes", len(self.class_names()))

    def __getitem__(self, batch_indexes_tup):
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_training:
            _transform = self._training_transforms(
                size=crop_size, ignore_idx=self.ignore_label
            )
        elif self.is_evaluation:
            _transform = self._evaluation_transforms(size=crop_size)
        else:
            _transform = self._validation_transforms(size=crop_size)

        coco = self.coco
        img_id = self.ids[img_index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]

        rgb_img = self.read_image_opencv(os.path.join(self.img_dir, path))
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        im_height, im_width = rgb_img.shape[:2]

        mask = self._gen_seg_mask(
            cocotarget, img_metadata["height"], img_metadata["width"]
        )

        data = {"image": rgb_img, "mask": None if self.is_evaluation else mask}

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        data["label"] = data["mask"]
        del data["mask"]

        if self.is_evaluation:
            img_name = path.replace("jpg", "png")
            data["file_name"] = img_name
            data["im_width"] = im_width
            data["im_height"] = im_height

        return data

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        coco_to_pascal = self.coco_to_pascal_mapping()
        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in coco_to_pascal:
                c = coco_to_pascal.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(
                    np.uint8
                )
        return mask

    def _training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        aug_list = [
            T.RandomResize(opts=self.opts),
            T.RandomCrop(opts=self.opts, size=size),
            T.RandomHorizontalFlip(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [T.Resize(opts=self.opts), T.ToTensor(opts=self.opts)]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.segmentation.resize_input_images", False):
            aug_list.append(T.Resize(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __len__(self):
        return len(self.ids)

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

    @staticmethod
    def coco_to_pascal_mapping():
        return [
            0,
            5,
            2,
            16,
            9,
            44,
            6,
            3,
            17,
            62,
            21,
            67,
            18,
            19,
            4,
            1,
            64,
            20,
            63,
            7,
            72,
        ]

    def __repr__(self):
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self._evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\t\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.ids),
            transforms_str,
        )

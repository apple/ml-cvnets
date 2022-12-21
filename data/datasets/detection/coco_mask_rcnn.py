#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from typing import Optional, Tuple, Dict, List
import math
import argparse

from .coco_base import COCODetection
from ...transforms import image_pil as T
from ...datasets import register_dataset
from ...collate_fns import register_collate_fn


@register_dataset(name="coco_mask_rcnn", task="detection")
class COCODetectionMaskRCNN(COCODetection):
    """Dataset class for the MS COCO Object Detection using Mask RCNN .

    Args:
        opts :
            Command line arguments
        is_training : bool
            A flag used to indicate training or validation mode
        is_evaluation : bool
            A flag used to indicate evaluation (or inference) mode
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

        # set the collate functions for the dataset
        setattr(opts, "dataset.collate_fn_name_train", "coco_mask_rcnn_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "coco_mask_rcnn_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "coco_mask_rcnn_collate_fn")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.detection.coco-mask-rcnn.use-lsj-aug",
            action="store_true",
            help="Use large scale jitter augmentation for training Mask RCNN model",
        )

        return parser

    def _training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        """Training data augmentation methods
        (Resize --> RandomHorizontalFlip --> ToTensor).
        """

        if getattr(self.opts, "dataset.detection.coco_mask_rcnn.use_lsj_aug", False):
            aug_list = [
                T.ScaleJitter(opts=self.opts),
                T.FixedSizeCrop(opts=self.opts),
                T.RandomHorizontalFlip(opts=self.opts),
                T.ToTensor(opts=self.opts),
            ]
        else:
            aug_list = [
                T.Resize(opts=self.opts, img_size=size),
                T.RandomHorizontalFlip(opts=self.opts),
                T.ToTensor(opts=self.opts),
            ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple, *args, **kwargs):
        """Implements validation transformation method (Resize --> ToTensor)."""
        aug_list = [
            T.Resize(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple, *args, **kwargs) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:  # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_name = self.get_image(image_id=image_id)
        im_width, im_height = image.size

        boxes, labels, mask = self.get_boxes_and_labels(
            image_id=image_id,
            image_width=im_width,
            image_height=im_height,
            include_masks=True,
        )

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes,
            "mask": mask,
        }

        if transform_fn is not None:
            data = transform_fn(data)

        output_data = {
            "samples": {
                "image": data["image"],
                # PyTorch Mask RCNN implementation expect labels as an input. Because we do not want to change the
                # the training infrastructure of CVNets library, we pass labels as part of image key and
                # handle it in the model.
                "label": {
                    "labels": data["box_labels"],
                    "boxes": data["box_coordinates"],
                    "masks": data["mask"],
                },
            },
            "targets": {
                "image_id": torch.tensor(image_id),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data


@register_collate_fn(name="coco_mask_rcnn_collate_fn")
def coco_mask_rcnn_collate_fn(batch: List, opts, *args, **kwargs) -> Dict:
    new_batch = {"samples": {"image": [], "label": []}, "targets": []}

    for b_id, batch_ in enumerate(batch):
        new_batch["samples"]["image"].append(batch_["samples"]["image"])
        new_batch["samples"]["label"].append(batch_["samples"]["label"])
        new_batch["targets"].append(batch_["targets"])

    return new_batch

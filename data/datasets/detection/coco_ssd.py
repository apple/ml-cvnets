#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from typing import Optional, Tuple, Dict
import math

from utils import logger
from cvnets.matcher_det import build_matcher
from cvnets.anchor_generator import build_anchor_generator

from .coco_base import COCODetection
from ...transforms import image_pil as T
from ...datasets import register_dataset
from ...collate_fns import register_collate_fn


@register_dataset(name="coco_ssd", task="detection")
class COCODetectionSSD(COCODetection):
    """Dataset class for the MS COCO Object Detection using Single Shot Object Detector (SSD).

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

        anchor_gen_name = getattr(opts, "anchor_generator.name", None)
        if anchor_gen_name is None or anchor_gen_name != "ssd":
            logger.error("For SSD, we need --anchor-generator.name to be ssd")

        self.anchor_box_generator = build_anchor_generator(opts=opts, is_numpy=True)

        self.output_strides = self.anchor_box_generator.output_strides

        if getattr(opts, "matcher.name") != "ssd":
            logger.error("For SSD, we need --matcher.name as ssd")

        self.match_prior = build_matcher(opts=opts)

        # set the collate functions for the dataset
        setattr(opts, "dataset.collate_fn_name_train", "coco_ssd_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "coco_ssd_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "coco_ssd_collate_fn")

    def _training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        """Training data augmentation methods
        (SSDCroping --> PhotometricDistort --> RandomHorizontalFlip -> Resize --> ToTensor).
        """
        aug_list = [
            T.SSDCroping(opts=self.opts),
            T.PhotometricDistort(opts=self.opts),
            T.RandomHorizontalFlip(opts=self.opts),
            T.Resize(opts=self.opts, img_size=size),
            T.BoxPercentCoords(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple, *args, **kwargs):
        """Implements validation transformation method (Resize --> ToTensor)."""
        aug_list = [
            T.Resize(opts=self.opts),
            T.BoxPercentCoords(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def generate_anchors(self, height, width):
        """Generate anchors **on-the-fly** based on the input resolution."""
        anchors = []
        for output_stride in self.output_strides:
            if output_stride == -1:
                fm_width = fm_height = 1
            else:
                fm_width = int(math.ceil(width / output_stride))
                fm_height = int(math.ceil(height / output_stride))
            fm_anchor = self.anchor_box_generator(
                fm_height=fm_height, fm_width=fm_width, fm_output_stride=output_stride
            )
            anchors.append(fm_anchor)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:
            # During evaluation, we use base class
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_fname = self.get_image(image_id=image_id)
        im_width, im_height = image.size
        boxes, labels = self.get_boxes_and_labels(
            image_id=image_id, image_width=im_width, image_height=im_height
        )

        data = {"image": image, "box_labels": labels, "box_coordinates": boxes}

        data = transform_fn(data)

        # convert to priors
        anchors = self.generate_anchors(height=crop_size_h, width=crop_size_w)

        gt_coordinates, gt_labels = self.match_prior(
            gt_boxes=data["box_coordinates"],
            gt_labels=data["box_labels"],
            anchors=anchors,
        )

        output_data = {
            "image": {"image": data["image"]},
            "label": {
                "box_labels": gt_labels,
                "box_coordinates": gt_coordinates,
                "image_id": torch.tensor(image_id),
                "image_width": torch.tensor(im_width),
                "image_height": torch.tensor(im_height),
            },
        }

        return output_data

    def __repr__(self):
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self._evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\ttransforms={}\n\tmatcher={}\n\tanchor_gen={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.ids),
            transforms_str,
            self.match_prior,
            self.anchor_box_generator,
        )


@register_collate_fn(name="coco_ssd_collate_fn")
def coco_ssd_collate_fn(batch, opts):

    new_batch = {"image": dict(), "label": dict()}

    for b_id, batch_ in enumerate(batch):
        # prepare inputs
        if "image" in batch_["image"]:
            if "image" in new_batch["image"]:
                new_batch["image"]["image"].append(batch_["image"]["image"])
            else:
                new_batch["image"]["image"] = [batch_["image"]["image"]]

        # prepare outputs
        if "box_labels" in batch_["label"]:
            if "box_labels" in new_batch["label"]:
                new_batch["label"]["box_labels"].append(batch_["label"]["box_labels"])
            else:
                new_batch["label"]["box_labels"] = [batch_["label"]["box_labels"]]

        if "box_coordinates" in batch_["label"]:
            if "box_coordinates" in new_batch["label"]:
                new_batch["label"]["box_coordinates"].append(
                    batch_["label"]["box_coordinates"]
                )
            else:
                new_batch["label"]["box_coordinates"] = [
                    batch_["label"]["box_coordinates"]
                ]

        if "image_id" in batch_["label"]:
            if "image_id" in new_batch["label"]:
                new_batch["label"]["image_id"].append(batch_["label"]["image_id"])
            else:
                new_batch["label"]["image_id"] = [batch_["label"]["image_id"]]

        if "image_width" in batch_["label"]:
            if "image_width" in new_batch["label"]:
                new_batch["label"]["image_width"].append(batch_["label"]["image_width"])
            else:
                new_batch["label"]["image_width"] = [batch_["label"]["image_width"]]

        if "image_height" in batch_["label"]:
            if "image_height" in new_batch["label"]:
                new_batch["label"]["image_height"].append(
                    batch_["label"]["image_height"]
                )
            else:
                new_batch["label"]["image_height"] = [batch_["label"]["image_height"]]

    # stack inputs
    new_batch["image"]["image"] = torch.stack(new_batch["image"]["image"], dim=0)

    # stack outputs
    if "box_labels" in new_batch["label"]:
        new_batch["label"]["box_labels"] = torch.stack(
            new_batch["label"]["box_labels"], dim=0
        )

    if "box_coordinates" in new_batch["label"]:
        new_batch["label"]["box_coordinates"] = torch.stack(
            new_batch["label"]["box_coordinates"], dim=0
        )

    if "image_id" in new_batch["label"]:
        new_batch["label"]["image_id"] = torch.stack(
            new_batch["label"]["image_id"], dim=0
        )

    if "image_width" in new_batch["label"]:
        new_batch["label"]["image_width"] = torch.stack(
            new_batch["label"]["image_width"], dim=0
        )

    if "image_height" in new_batch["label"]:
        new_batch["label"]["image_height"] = torch.stack(
            new_batch["label"]["image_height"], dim=0
        )

    return new_batch

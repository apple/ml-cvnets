#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import math
from typing import Dict, Tuple

import torch
import torch.utils.data as data

from cvnets.anchor_generator import build_anchor_generator
from cvnets.matcher_det import build_matcher


class DummySSDDetectionDataset(data.Dataset):
    """
    Dummy SSD Detection Dataset for CI/CD testing

    Args:
        opts: command-line arguments

    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.dummy_image = torch.randint(low=0, high=255, size=(300, 500, 3))

        self.anchor_box_generator = build_anchor_generator(opts=opts, is_numpy=True)

        self.output_strides = self.anchor_box_generator.output_strides

        if getattr(opts, "matcher.name") != "ssd":
            ValueError("For SSD, we need --matcher.name as ssd")

        self.match_prior = build_matcher(opts=opts)

        self.n_classes = 80
        # set the collate functions for the dataset
        setattr(opts, "dataset.collate_fn_name_train", "coco_ssd_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "coco_ssd_collate_fn")
        setattr(opts, "dataset.collate_fn_name_test", "coco_ssd_collate_fn")

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
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        input_img = torch.randint(low=0, high=255, size=(3, crop_size_h, crop_size_w))
        input_img = input_img.float().div(255.0)

        n_boxes = 4
        min_dim = min(crop_size_h, crop_size_h)

        boxes_x = torch.randint(
            low=0, high=crop_size_w // 4, size=(n_boxes,), dtype=torch.float
        )
        boxes_y = torch.randint(
            low=1, high=crop_size_h // 4, size=(n_boxes,), dtype=torch.float
        )
        boxes_w = torch.randint(
            low=crop_size_w // 2, high=crop_size_w, size=(n_boxes,), dtype=torch.float
        )
        boxes_h = torch.randint(
            low=crop_size_h // 2, high=crop_size_h, size=(n_boxes,), dtype=torch.float
        )

        boxes = torch.stack([boxes_x, boxes_y, boxes_w, boxes_h])

        boxes[:, 0::2] /= crop_size_w
        boxes[:, 1::2] /= crop_size_h

        labels = torch.randint(low=1, high=self.n_classes, size=(n_boxes,)).long()

        # convert to priors
        anchors = self.generate_anchors(height=crop_size_h, width=crop_size_w)

        gt_coordinates, gt_labels = self.match_prior(
            gt_boxes=boxes,
            gt_labels=labels,
            anchors=anchors,
        )

        # Make sure there are no NaNs in the coordinates
        gt_coordinates[gt_coordinates.isnan()] = 0

        return {
            "samples": {"image": input_img},
            "targets": {
                "box_labels": gt_labels,
                "box_coordinates": gt_coordinates,
                "image_id": torch.randint(low=0, high=1000, size=(1,)).long(),
                "image_width": torch.tensor(crop_size_w),
                "image_height": torch.tensor(crop_size_h),
            },
        }

    def __len__(self) -> int:
        return 10

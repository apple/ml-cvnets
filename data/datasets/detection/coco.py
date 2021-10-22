#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from pycocotools.coco import COCO
import os
from typing import Optional, Tuple, Dict
import numpy as np
import math

from utils import logger
from cvnets.misc.anchor_generator import SSDAnchorGenerator
from cvnets.misc.match_prior import SSDMatcher

from ...transforms import image as tf
from ...datasets import BaseImageDataset, register_dataset


COCO_CLASS_LIST = ['background',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush'
                   ]


@register_dataset(name="coco", task="detection")
class COCODetection(BaseImageDataset):
    """
        Dataset class for the COCO Object detection

        Dataset structure should be something like this
        + coco
        + --- annotations
        + ------ *.json
        + --- images
        + ------ train2017
        + ---------- *.jpg
        + ------ val2017
        + ---------- *.jpg

    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        super(COCODetection, self).__init__(opts=opts, is_training=is_training, is_evaluation=is_evaluation)

        split = 'train' if is_training else 'val'
        year = 2017
        ann_file = os.path.join(self.root, 'annotations/instances_{}{}.json'.format(split, year))
        self.coco = COCO(ann_file)
        self.img_dir = os.path.join(self.root, 'images/{}{}'.format(split, year))
        self.ids = list(self.coco.imgToAnns.keys()) if is_training else list(self.coco.imgs.keys())

        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}
        self.num_classes = len(COCO_CLASS_LIST)

        setattr(opts, "model.detection.n_classes", self.num_classes)

        assert len(self.contiguous_id_to_coco_id.keys()) + 1 == self.num_classes  # +1 for background

    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        # implement these functions in sub classes
        raise NotImplementedError

    def validation_transforms(self, size: tuple, *args, **kwargs):
        raise NotImplementedError

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.detection.resize_input_images", False):
            aug_list.append(tf.Resize(opts=self.opts, size=size))

        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        elif self.is_evaluation:
            transform_fn = self.evaluation_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_name = self._get_image(image_id=image_id)
        boxes, labels = self._get_annotation(image_id=image_id)

        im_height, im_width = image.shape[:2]

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes
        }

        if transform_fn is not None:
            data = transform_fn(data)

        new_data = {
            "image": data["image"],
            "label": {
                "box_labels": data["box_labels"],
                "box_coordinates": data["box_coordinates"],
                "image_id": image_id
            }
        }

        del data

        if self.is_evaluation:
            new_data["file_name"] = img_name
            new_data["im_width"] = im_width
            new_data["im_height"] = im_height

        return new_data

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)

        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))

        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def _get_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.img_dir, file_name)
        image = self.read_image(image_file)
        return image, file_name


@register_dataset(name="coco_ssd", task="detection")
class COCODetectionSSD(COCODetection):
    """
        Dataset class for the COCO Object detection using SSD
    """
    def __init__(self, opts, is_training: Optional[bool] = True, is_evaluation: Optional[bool] = False):
        super(COCODetectionSSD, self).__init__(
            opts=opts,
            is_training=is_training,
            is_evaluation=is_evaluation
        )

        anchors_aspect_ratio = getattr(opts, "model.detection.ssd.anchors_aspect_ratio", [[2, 3], [2, 3], [2]])
        output_strides = getattr(opts, "model.detection.ssd.output_strides", [8, 16, 32])

        if len(anchors_aspect_ratio) != len(output_strides):
            logger.error(
                "SSD model requires anchors to be defined for feature maps from each output stride. So,"
                "len(anchors_per_location) == len(output_strides). "
                "Got len(output_strides)={} and len(anchors_aspect_ratio)={}. "
                "Please specify correct arguments using following arguments: "
                "\n--model.detection.ssd.anchors-aspect-ratio "
                "\n--model.detection.ssd.output-strides".format(
                    len(output_strides),
                    len(anchors_aspect_ratio),
                )
            )

        self.output_strides = output_strides

        self.anchor_box_generator = SSDAnchorGenerator(
            output_strides=output_strides,
            aspect_ratios=anchors_aspect_ratio,
            min_ratio=getattr(opts, "model.detection.ssd.min_box_size", 0.1),
            max_ratio=getattr(opts, "model.detection.ssd.max_box_size", 1.05)
        )

        self.match_prior = SSDMatcher(
            center_variance=getattr(opts, "model.detection.ssd.center_variance", 0.1),
            size_variance=getattr(opts, "model.detection.ssd.size_variance", 0.2),
            iou_threshold=getattr(opts, "model.detection.ssd.iou_threshold", 0.5) # we use nms_iou_threshold during inference
        )

    def training_transforms(self, size: tuple, ignore_idx: Optional[int] = 255):
        aug_list = [
            #tf.RandomZoomOut(opts=self.opts),
            tf.SSDCroping(opts=self.opts),
            tf.PhotometricDistort(opts=self.opts),
            tf.RandomHorizontalFlip(opts=self.opts),
            tf.BoxPercentCoords(opts=self.opts),
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]

        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [
            tf.BoxPercentCoords(opts=self.opts),
            tf.Resize(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ]
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def evaluation_transforms(self, size: tuple, *args, **kwargs):
        return self.validation_transforms(size=size)

    def get_anchors(self, crop_size_h, crop_size_w):
        anchors = []
        for output_stride in self.output_strides:
            if output_stride == -1:
                fm_width = fm_height = 1
            else:
                fm_width = int(math.ceil(crop_size_w / output_stride))
                fm_height = int(math.ceil(crop_size_h / output_stride))
            fm_anchor = (
                self.anchor_box_generator(
                    fm_height=fm_height,
                    fm_width=fm_width,
                    fm_output_stride=output_stride
                )
            )
            anchors.append(fm_anchor)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        image_id = self.ids[img_index]

        image, img_fname = self._get_image(image_id=image_id)
        boxes, labels = self._get_annotation(image_id=image_id)

        data = {
            "image": image,
            "box_labels": labels,
            "box_coordinates": boxes
        }
        data = transform_fn(data)

        # convert to priors
        anchors = self.get_anchors(crop_size_h=crop_size_h, crop_size_w=crop_size_w)

        gt_coordinates, gt_labels = self.match_prior(
            gt_boxes_cor=data["box_coordinates"],
            gt_labels=data["box_labels"],
            reference_boxes_ctr=anchors
        )

        return {
            "image": data["image"],
            "label": {
                "box_labels": gt_labels,
                "box_coordinates": gt_coordinates
            }
        }

    def __repr__(self):
        from utils.tensor_utils import tensor_size_from_opts
        im_h, im_w = tensor_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self.training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self.evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self.validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\t is_training={}\n\tsamples={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            len(self.ids),
            transforms_str
        )

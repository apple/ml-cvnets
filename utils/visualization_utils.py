#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import cv2
import numpy as np
import copy
from typing import Optional, List

from utils.color_map import Colormap
from utils import logger

FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2


def visualize_boxes_xyxy(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Utility function to draw bounding boxes of objects on a given image"""
    boxes = boxes.astype(np.int)

    new_image = copy.deepcopy(image)
    for box_idx in range(boxes.shape[0]):
        coords = boxes[box_idx]
        r, g, b = 255, 0, 0
        # top -left corner
        start_coord = (coords[0], coords[1])
        # bottom-right corner
        end_coord = (coords[2], coords[3])
        cv2.rectangle(new_image, end_coord, start_coord, (r, g, b), thickness=1)
    return new_image


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    color_map: Optional = None,
    object_names: Optional[List] = None,
    is_bgr_format: Optional[bool] = False,
    save_path: Optional[str] = None,
) -> None:
    """Utility function to draw bounding boxes of objects along with their labels and score on a given image"""
    boxes = boxes.astype(np.int)

    if is_bgr_format:
        # convert from BGR to RGB colorspace
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if color_map is None:
        color_map = Colormap().get_box_color_codes()

    for label, score, coords in zip(labels, scores, boxes):
        r, g, b = color_map[label]
        c1 = (coords[0], coords[1])
        c2 = (coords[2], coords[3])

        cv2.rectangle(image, c1, c2, (r, g, b), thickness=RECT_BORDER_THICKNESS)
        if object_names is not None:
            label_text = "{label}: {score:.2f}".format(
                label=object_names[label], score=score
            )
            t_size = cv2.getTextSize(label_text, FONT_SIZE, 1, TEXT_THICKNESS)[0]
            new_c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(image, c1, new_c2, (r, g, b), -1)
            cv2.putText(
                image,
                label_text,
                (c1[0], c1[1] + t_size[1] + 4),
                FONT_SIZE,
                1,
                LABEL_COLOR,
                TEXT_THICKNESS,
            )

    if save_path is not None:
        cv2.imwrite(save_path, image)
        logger.log("Detection results stored at: {}".format(save_path))
    return image


def convert_to_cityscape_format(img: Tensor) -> Tensor:
    """Utility to map predicted segmentation labels to cityscapes format"""
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img

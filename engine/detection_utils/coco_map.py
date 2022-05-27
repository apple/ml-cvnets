#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from typing import Optional, List
from contextlib import redirect_stdout
import io

from utils import logger


def coco_evaluation(
    opts,
    predictions: List[np.ndarray],
    split: Optional[str] = "val",
    year: Optional[int] = 2017,
    iou_type: Optional[str] = "bbox",
    *args,
    **kwargs
) -> None:
    coco_results = []
    root = getattr(opts, "dataset.root_val", None)
    ann_file = os.path.join(root, "annotations/instances_{}{}.json".format(split, year))
    coco = COCO(ann_file)

    coco_categories = sorted(coco.getCatIds())
    coco_id_to_contiguous_id = {
        coco_id: i + 1 for i, coco_id in enumerate(coco_categories)
    }
    contiguous_id_to_coco_id = {v: k for k, v in coco_id_to_contiguous_id.items()}

    for i, (image_id, boxes, labels, scores) in enumerate(predictions):
        if labels.shape[0] == 0:
            continue

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": contiguous_id_to_coco_id[labels[k]],
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1],
                    ],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    if len(coco_results) == 0:
        logger.error("Cannot compute COCO stats. Please check the predictions")

    with redirect_stdout(io.StringIO()):
        coco_dt = COCO.loadRes(coco, coco_results)

    # Run COCO evaluation
    coco_eval = COCOeval(coco, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def compute_quant_scores(opts, predictions: List, *args, **kwargs) -> None:
    dataset_name = getattr(opts, "dataset.name", None)
    if dataset_name.find("coco") > -1:
        coco_evaluation(opts=opts, predictions=predictions)
    else:
        raise NotImplementedError

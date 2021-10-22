#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from typing import Optional, List


def coco_evaluation(opts,
                    predictions: List[np.ndarray],
                    output_dir: Optional[str] = "coco_eval_results",
                    split: Optional[str] = 'val',
                    year: Optional[int] = 2017,
                    iou_type: Optional[str] = "bbox") -> None:
    coco_results = []
    root = getattr(opts, "dataset.root_val", None)
    ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
    coco = COCO(ann_file)

    coco_categories = sorted(coco.getCatIds())
    coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
    contiguous_id_to_coco_id = {v: k for k, v in coco_id_to_contiguous_id.items()}

    ids = list(coco.imgs.keys())

    for i, (img_idx, boxes, labels, scores) in enumerate(predictions):
        image_id = ids[img_idx]
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
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    json_result_file = os.path.join(output_dir, iou_type + ".json")

    if os.path.isfile(json_result_file):
        # delete the json file if it exists
        os.remove(json_result_file)

    with open(json_result_file, "w") as f:
        # write results to the JSON file
        json.dump(coco_results, f)

    # Run COCO evaluation
    coco_dt = coco.loadRes(json_result_file)
    coco_eval = COCOeval(coco, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def compute_quant_scores(opts,
                         predictions: List,
                         output_dir: Optional[str] = "coco_eval_results", *args, **kwargs) -> None:
    dataset_name = getattr(opts, "dataset.name", None)
    if dataset_name.find("coco") > -1:
        coco_evaluation(opts=opts, predictions=predictions, output_dir=output_dir)
    else:
        raise NotImplementedError
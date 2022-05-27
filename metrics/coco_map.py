#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from typing import Optional, Tuple, Any, Dict, List
import io
import os
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from contextlib import redirect_stdout

from cvnets.models.detection.base_detection import DetectionPredTuple
from utils.tensor_utils import all_gather_list
from utils import logger
from utils.ddp_utils import is_master

from . import register_stats_fn


@register_stats_fn(name="coco_map")
class COCOEvaluator(object):
    def __init__(
        self,
        opts,
        iou_types: Optional[List] = ["bbox"],
        split: Optional[str] = "val",
        year: Optional[int] = 2017,
        use_distributed: Optional[bool] = False,
        *args,
        **kwargs
    ):
        # disable printing on console, so that pycocotools print statements are not printed on console
        logger.disable_printing()

        root = getattr(opts, "dataset.root_val", None)
        ann_file = os.path.join(
            root, "annotations/instances_{}{}.json".format(split, year)
        )
        coco_gt = COCO(ann_file)

        coco_categories = sorted(coco_gt.getCatIds())
        coco_id_to_contiguous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(coco_categories)
        }
        self.contiguous_id_to_coco_id = {
            v: k for k, v in coco_id_to_contiguous_id.items()
        }

        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.use_distributed = use_distributed
        self.is_master_node = is_master(opts)

        self.coco_results = {iou_type: [] for iou_type in iou_types}

        # enable printing, to enable cvnets log printing
        logger.enable_printing()

    def prepare_predictions(self, predictions: Dict, targets: Dict):
        if not (
            isinstance(predictions, Dict)
            and ({"detections"} <= set(list(predictions.keys())))
        ):
            logger.error(
                "For coco evaluation during training, the output from the model should be a dictionary "
                "and should contain the results in a key called detections"
            )

        detections = predictions["detections"]
        if isinstance(detections, List) and isinstance(
            detections[0], DetectionPredTuple
        ):
            self.prepare_cache_results(
                detection_results=detections,
                image_ids=targets["image_id"],
                image_widths=targets["image_width"],
                image_heights=targets["image_height"],
                iou_type="bbox",
            )
        elif isinstance(detections, DetectionPredTuple):
            self.prepare_cache_results(
                detection_results=[detections],  # create a list
                image_ids=targets["image_id"],
                image_widths=targets["image_width"],
                image_heights=targets["image_height"],
                iou_type="bbox",
            )
        else:
            logger.error(
                "For coco evaluation during training, the results should be stored as a List of DetectionPredTuple"
            )

    def prepare_cache_results(
        self, detection_results: List, image_ids, image_widths, image_heights, iou_type
    ):
        batch_results = []
        for detection_result, img_id, img_w, img_h in zip(
            detection_results, image_ids, image_widths, image_heights
        ):
            label = detection_result.labels

            if label.numel() == 0:
                # no detections
                continue
            box = detection_result.boxes
            score = detection_result.scores

            img_id, img_w, img_h = img_id.item(), img_w.item(), img_h.item()

            box[..., 0::2] = torch.clip(box[..., 0::2] * img_w, min=0, max=img_w)
            box[..., 1::2] = torch.clip(box[..., 1::2] * img_h, min=0, max=img_h)

            # convert box from xyxy to xywh format
            box[..., 2] = box[..., 2] - box[..., 0]
            box[..., 3] = box[..., 3] - box[..., 1]

            box = box.cpu().numpy()
            label = label.cpu().numpy()
            score = score.cpu().numpy()

            batch_results.extend(
                [
                    {
                        "image_id": img_id,
                        "category_id": self.contiguous_id_to_coco_id[label[bbox_id]],
                        "bbox": box[bbox_id].tolist(),
                        "score": score[bbox_id],
                    }
                    for bbox_id in range(box.shape[0])
                    if label[bbox_id] > 0
                ]
            )

        self.coco_results[iou_type].extend(batch_results)

    def gather_coco_results(self):

        # synchronize results across different devices
        for iou_type, coco_results in self.coco_results.items():
            # agg_coco_results as List[List].
            # The outer list is for processes and inner list is for coco_results in the process
            if self.use_distributed:
                agg_coco_results = all_gather_list(coco_results)

                merged_coco_results = []
                # filter the duplicates
                for (
                    p_coco_results
                ) in agg_coco_results:  # retrieve results from each process
                    merged_coco_results.extend(p_coco_results)
            else:
                merged_coco_results = coco_results

            self.coco_results[iou_type] = merged_coco_results

    def summarize_coco_results(self) -> Dict:

        logger.disable_printing()

        stats_map = dict()
        for iou_type, coco_results in self.coco_results.items():
            if len(coco_results) < 1:
                # during initial epochs, we may not have any sample results, so we can skip this part
                map_val = 0.0
            else:
                try:
                    with redirect_stdout(io.StringIO()):
                        coco_dt = COCO.loadRes(self.coco_gt, coco_results)

                    coco_eval = COCOeval(
                        cocoGt=self.coco_gt, cocoDt=coco_dt, iouType=iou_type
                    )
                    coco_eval.evaluate()
                    coco_eval.accumulate()

                    if self.is_master_node:
                        logger.enable_printing()

                    logger.log("Results for iouType={}".format(iou_type))
                    coco_eval.summarize()
                    map_val = coco_eval.stats[0].item()
                except Exception as e:
                    map_val = 0.0
            stats_map[iou_type] = map_val * 100

        logger.enable_printing()
        return stats_map

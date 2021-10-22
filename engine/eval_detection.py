#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy
import os.path
import numpy as np
import torch
import multiprocessing
from torch.cuda.amp import autocast
from torch.nn import functional as F
import cv2
from tqdm import tqdm
import glob
from typing import Optional
from torch import Tensor, nn


from common import SUPPORTED_IMAGE_EXTNS
from options.opts import get_detection_eval_arguments
from cvnets import get_model
from cvnets.models.detection.ssd import DetectionPredTuple
from data import create_eval_loader
from data.datasets.dataset_base import BaseImageDataset
from data.datasets.detection.coco import COCO_CLASS_LIST as object_names
from utils.tensor_utils import to_numpy, tensor_size_from_opts
from utils.color_map import Colormap
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master
from utils import logger
from engine.utils import print_summary
from engine.detection_utils.coco_map import compute_quant_scores


FONT_SIZE = cv2.FONT_HERSHEY_PLAIN
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2
COLOR_MAP = Colormap().get_box_color_codes()


def predict_and_save(opts,
                     input_tensor: Tensor,
                     model: nn.Module,
                     input_arr: Optional[np.ndarray] = None,
                     device: Optional = torch.device("cpu"),
                     mixed_precision_training: Optional[bool] = False,
                     is_validation: Optional[bool] = False,
                     file_name: Optional[str] = None,
                     output_stride: Optional[int] = 32, # Default is 32 because ImageNet models have 5 downsampling stages (2^5 = 32)
                     orig_h: Optional[int] = None,
                     orig_w: Optional[int] = None
                     ):

    if input_arr is None and not is_validation:
        input_arr = (
            to_numpy(input_tensor) # convert to numpy
            .squeeze(0) # remove batch dimension
        )

    curr_height, curr_width = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_height // output_stride) * output_stride
    new_w = (curr_width // output_stride) * output_stride

    if new_h != curr_height or new_w != curr_width:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # move data to device
    input_tensor = input_tensor.to(device)

    with autocast(enabled=mixed_precision_training):
        # prediction
        prediction: DetectionPredTuple = model.predict(input_tensor, is_scaling=False)

    # convert tensors to boxes
    boxes = prediction.boxes.cpu().numpy()
    labels = prediction.labels.cpu().numpy()
    scores = prediction.scores.cpu().numpy()

    if orig_w is None:
        assert orig_h is None
        orig_h, orig_w = input_arr.shape[:2]
    elif orig_h is None:
        assert orig_w is None
        orig_h, orig_w = input_arr.shape[:2]

    assert orig_h is not None and orig_w is not None
    boxes[..., 0::2] = boxes[..., 0::2] * orig_w
    boxes[..., 1::2] = boxes[..., 1::2] * orig_h
    boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
    boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

    if is_validation:
        return boxes, labels, scores

    boxes = boxes.astype(np.int)

    for label, score, coords in zip(labels, scores, boxes):
        r, g, b = COLOR_MAP[label]
        c1 = (coords[0], coords[1])
        c2 = (coords[2], coords[3])

        cv2.rectangle(input_arr, c1, c2, (r, g, b), thickness=RECT_BORDER_THICKNESS)
        label_text = '{label}: {score:.2f}'.format(label=object_names[label], score=score)
        t_size = cv2.getTextSize(label_text, FONT_SIZE, 1, TEXT_THICKNESS)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(input_arr, c1, c2, (r, g, b), -1)
        cv2.putText(input_arr, label_text, (c1[0], c1[1] + t_size[1] + 4), FONT_SIZE, 1, LABEL_COLOR, TEXT_THICKNESS)

    if file_name is not None:
        file_name = file_name.split(os.sep)[-1].split(".")[0] + ".jpg"
        res_dir = "{}/detection_results".format(getattr(opts, "common.exp_loc", None))
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        res_fname = "{}/{}".format(res_dir, file_name)
        cv2.imwrite(res_fname, input_arr)
        logger.log("Detection results stored at: {}".format(res_fname))


def predict_labeled_dataset(opts, **kwargs):
    device = getattr(opts, "dev.device", torch.device('cpu'))

    # set-up data loaders
    val_loader = create_eval_loader(opts)

    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning('Model is in training mode. Switching to evaluation mode')
        model.eval()

    mixed_precision_training = getattr(opts, "common.mixed_precision", False)

    with torch.no_grad():
        predictions = {}
        for img_idx, batch in tqdm(enumerate(val_loader)):
            input_img, target_label = batch['image'], batch['label']

            batch_size = input_img.shape[0]
            assert batch_size == 1, "We recommend to run segmentation evaluation with a batch size of 1"

            orig_w = batch["im_width"][0].item()
            orig_h = batch["im_height"][0].item()

            boxes, labels, scores = predict_and_save(
                opts=opts,
                input_tensor=input_img,
                model=model,
                device=device,
                mixed_precision_training=mixed_precision_training,
                is_validation=True,
                orig_w=orig_w,
                orig_h=orig_h
            )

            predictions[img_idx] = (img_idx, boxes, labels, scores)
        predictions = [predictions[i] for i in predictions.keys()]

        compute_quant_scores(opts=opts, predictions=predictions)


def predict_image(opts, image_fname, **kwargs):
    if not os.path.isfile(image_fname):
        logger.error("Image file does not exist at: {}".format(image_fname))

    input_img = BaseImageDataset.read_image(path=image_fname)
    input_imp_copy = copy.deepcopy(input_img)
    orig_h, orig_w = input_imp_copy.shape[:2]

    # Resize the image to the resolution that detector supports
    res_h, res_w = tensor_size_from_opts(opts)
    input_img = cv2.resize(input_img, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

    # HWC --> CHW
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = (
        torch.div(
            torch.from_numpy(input_img).float(), # convert to float tensor
            255.0 # convert from [0, 255] to [0, 1]
        ).unsqueeze(dim=0) # add a dummy batch dimension
    )

    image_fname = image_fname.split(os.sep)[-1]

    device = getattr(opts, "dev.device", torch.device('cpu'))
    mixed_precision_training = getattr(opts, "common.mixed_precision", False)
    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning('Model is in training mode. Switching to evaluation mode')
        model.eval()

    with torch.no_grad():
        predict_and_save(
            opts=opts,
            input_tensor=input_img,
            input_arr=input_imp_copy,
            file_name=image_fname,
            model=model,
            device=device,
            mixed_precision_training=mixed_precision_training,
            is_validation=False,
            orig_h=orig_h,
            orig_w=orig_w
        )


def predict_images_in_folder(opts, **kwargs):
    img_folder_path = getattr(opts, "evaluation.detection.path", None)
    if img_folder_path is None:
        logger.error("Image folder is not passed. Please use --evaluation.detection.path as an argument to pass the location of image folder".format(img_folder_path))
    elif not os.path.isdir(img_folder_path):
        logger.error("Image folder does not exist at: {}. Please check".format(img_folder_path))

    img_files = []
    for e in SUPPORTED_IMAGE_EXTNS:
        img_files_with_extn = glob.glob("{}/*{}".format(img_folder_path, e))
        if len(img_files_with_extn) > 0 and isinstance(img_files_with_extn, list):
            img_files.extend(img_files_with_extn)

    if len(img_files) == 0:
        logger.error("Number of image files found at {}: {}".format(img_folder_path, len(img_files)))

    logger.log("Number of image files found at {}: {}".format(img_folder_path, len(img_files)))

    device = getattr(opts, "dev.device", torch.device('cpu'))
    mixed_precision_training = getattr(opts, "common.mixed_precision", False)
    # set-up the model
    model = get_model(opts)
    model.eval()
    model = model.to(device=device)
    print_summary(opts=opts, model=model)

    if model.training:
        logger.warning('Model is in training mode. Switching to evaluation mode')
        model.eval()

    with torch.no_grad():
        for img_idx, image_fname in enumerate(img_files):
            input_img = BaseImageDataset.read_image(path=image_fname)
            input_imp_copy = copy.deepcopy(input_img)
            orig_h, orig_w = input_imp_copy.shape[:2]

            # Resize the image to the resolution that detector supports
            res_h, res_w = tensor_size_from_opts(opts)
            input_img = cv2.resize(input_img, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

            # HWC --> CHW
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = (
                torch.div(
                    torch.from_numpy(input_img).float(),  # convert to float tensor
                    255.0  # convert from [0, 255] to [0, 1]
                ).unsqueeze(dim=0)  # add a dummy batch dimension
            )

            image_fname = image_fname.split(os.sep)[-1]

            predict_and_save(
                opts=opts,
                input_tensor=input_img,
                input_arr=input_imp_copy,
                file_name=image_fname,
                model=model,
                device=device,
                mixed_precision_training=mixed_precision_training,
                is_validation=False,
                orig_h=orig_h,
                orig_w=orig_w
            )


def main_detection_evaluation(**kwargs):
    opts = get_detection_eval_arguments()

    dataset_name = getattr(opts, "dataset.name", "imagenet")
    if dataset_name.find("coco") > -1:
        # replace model specific datasets (e.g., coco_ssd) with general COCO dataset
        setattr(opts, "dataset.name", "coco")

    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    logger.log("Results (if any) will be stored here: {}".format(exp_dir))

    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    if num_gpus < 2:
        cls_norm_type = getattr(opts, "model.normalization.name", "batch_norm_2d")
        if cls_norm_type.find("sync") > -1:
            # replace sync_batch_norm with standard batch norm on PU
            setattr(opts, "model.normalization.name", cls_norm_type.replace("sync_", ""))
            setattr(opts, "model.classification.normalization.name", cls_norm_type.replace("sync_", ""))

    # we disable the DDP setting for evaluation tasks
    setattr(opts, "ddp.use_distributed", False)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if dataset_workers == -1:
        setattr(opts, "dataset.workers", n_cpus)

    # We are not performing any operation like resizing and cropping on images
    # Because image dimensions are different, we process 1 sample at a time.
    setattr(opts, "dataset.train_batch_size0", 1)
    setattr(opts, "dataset.val_batch_size0", 1)
    setattr(opts, "dev.device_id", None)

    eval_mode = getattr(opts, "evaluation.detection.mode", None)

    if eval_mode == "single_image":
        num_classes = getattr(opts, "model.detection.n_classes", 81)
        assert num_classes is not None

        # test a single image
        img_f_name = getattr(opts, "evaluation.detection.path", None)
        predict_image(opts, img_f_name, **kwargs)
    elif eval_mode == "image_folder":
        num_seg_classes = getattr(opts, "model.detection.n_classes", 81)
        assert num_seg_classes is not None

        # test all images in a folder
        predict_images_in_folder(opts=opts, **kwargs)
    elif eval_mode == "validation_set":
        # evaluate and compute stats for labeled image dataset
        # This is useful for generating results for validation set and compute quantitative results
        predict_labeled_dataset(opts=opts, **kwargs)
    else:
        logger.error(
            "Supported modes are single_image, image_folder, and validation_set. Got: {}".format(eval_mode)
        )


if __name__ == "__main__":
    main_detection_evaluation()
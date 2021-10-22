#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os.path
import numpy as np
import torch
import multiprocessing
import os
import cv2
from tqdm import tqdm
import glob
from typing import Optional, Tuple
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision

from utils import logger
from utils.tensor_utils import to_numpy, tensor_size_from_opts
from options.opts import get_segmentation_eval_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master
from cvnets import get_model
from data import create_eval_loader
from utils.color_map import Colormap
from engine.utils import print_summary
from common import SUPPORTED_IMAGE_EXTNS
from data.datasets.dataset_base import BaseImageDataset

"""
Notes:

1) We have separate scripts for evaluating segmentation models because the size of input images varies and
we do not want to apply any resizing operations to input because that distorts the quality and hurts the performance.

2) [Optional] We want to save the outputs in the same size as that of the input image.
"""


class ConfusionMatrix(object):
    """
        This class is based on FCN
            https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
    """
    def __init__(self):
        self.confusion_mat = None

    def update(self, ground_truth, prediction, n_classes):
        if self.confusion_mat is None:
            self.confusion_mat = torch.zeros((n_classes, n_classes), dtype=torch.int64, device=ground_truth.device)
        with torch.no_grad():
            k = (ground_truth >= 0) & (ground_truth < n_classes)
            inds = n_classes * ground_truth[k].to(torch.int64) + prediction[k]
            self.confusion_mat += torch.bincount(inds, minlength=n_classes ** 2).reshape(n_classes, n_classes)

    def reset(self):
        if self.confusion_mat is not None:
            self.confusion_mat.zero_()

    def compute(self):
        if self.confusion_mat is None:
            print("Confusion matrix is None. Check code")
            return None
        h = self.confusion_mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


def convert_to_cityscape_format(img):
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


def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np


def predict_and_save(opts,
                     input_tensor: Tensor,
                     file_name: str,
                     orig_h: int,
                     orig_w: int,
                     model: nn.Module,
                     target_label: Optional[Tensor] = None,
                     device: Optional = torch.device("cpu"),
                     mixed_precision_training: Optional[bool] = False,
                     confmat: Optional[ConfusionMatrix] = None,
                     cmap: list = Colormap().get_color_map_list(),
                     orig_image: Optional[np.ndarray] = None,
                     ):
    output_stride = getattr(opts, "model.segmentation.output_stride", 16)
    if output_stride == 1:
        output_stride = 32 # we set it to 32 because ImageNet models have 5 downsampling stages (2^5 = 32)

    #input_img_np = to_numpy(input_tensor).squeeze(0)  # remove the batch dimension

    curr_h, curr_w = input_tensor.shape[2:]

    # check if dimensions are multiple of output_stride, otherwise, we get dimension mismatch errors.
    # if not, then resize them
    new_h = (curr_h // output_stride) * output_stride
    new_w = (curr_w // output_stride) * output_stride

    if new_h != curr_h or new_w != curr_w:
        # resize the input image, so that we do not get dimension mismatch errors in the forward pass
        input_tensor = F.interpolate(input=input_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    file_name = file_name.split(os.sep)[-1].split(".")[0] + ".png"

    # move data to device
    input_tensor = input_tensor.to(device)
    if target_label is not None:
        target_label = target_label.to(device)

    with autocast(enabled=mixed_precision_training):
        # prediction
        pred_label = model(input_tensor)

    if isinstance(pred_label, Tuple):
        pred_mask = pred_label[0]
    elif isinstance(pred_label, Tensor):
        pred_mask = pred_label
    else:
        raise NotImplementedError
    pred_h, pred_w = pred_mask.shape[2:]
    if pred_h != orig_h or pred_w != orig_w:
        pred_mask = F.interpolate(input=pred_mask, size=(orig_h, orig_w), mode="nearest")

    num_classes = pred_mask.shape[1]
    pred_mask = (
        pred_mask
            .argmax(1)  # get the predicted label index
            .squeeze(0)  # remove the batch dimension
    )
    if target_label is not None and confmat is not None:
        confmat.update(ground_truth=target_label.flatten(), prediction=pred_mask.flatten(), n_classes=num_classes)

    if getattr(opts, "evaluation.segmentation.apply_color_map", False):
        pred_mask_pil = F_vision.to_pil_image(pred_mask.byte())
        pred_mask_pil.putpalette(cmap)
        pred_mask_pil = pred_mask_pil.convert('RGB')

        color_mask_dir = "{}/predictions_cmap".format(getattr(opts, "common.exp_loc", None))
        if not os.path.isdir(color_mask_dir):
            os.makedirs(color_mask_dir, exist_ok=True)
        color_mask_f_name = "{}/{}".format(color_mask_dir, file_name)
        pred_mask_pil.save(color_mask_f_name)

        if getattr(opts, "evaluation.segmentation.save_overlay_rgb_pred", False) \
                and isinstance(orig_image, np.ndarray) \
                and orig_image.ndim == 3: # Need RGB Image
            pred_mask_pil_np = np.array(pred_mask_pil)
            pred_mask_pil_np = cv2.cvtColor(pred_mask_pil_np, cv2.COLOR_RGB2BGR)

            mask_wt = getattr(opts, "evaluation.segmentation.overlay_mask_weight", 0.5)
            overlayed_img = cv2.addWeighted(orig_image, 1.0 - mask_wt, pred_mask_pil_np, mask_wt, 0)

            overlay_mask_dir = "{}/predictions_overlay".format(getattr(opts, "common.exp_loc", None))
            if not os.path.isdir(overlay_mask_dir):
                os.makedirs(overlay_mask_dir, exist_ok=True)
            overlay_mask_f_name = "{}/{}".format(overlay_mask_dir, file_name)

            cv2.imwrite(overlay_mask_f_name, overlayed_img)
        else:
            logger.warning(
                "For overlaying segmentation mask on RGB image, we need original image (shape=[H,W,C]) as "
                "an instance of np.ndarray. Got: {}".format(orig_image)
            )

    is_city_dataset = (getattr(opts, "dataset.name", "") == "cityscapes")
    if getattr(opts, "evaluation.segmentation.save_masks", False) or is_city_dataset:
        no_color_mask_dir = "{}/predictions_no_cmap".format(getattr(opts, "common.exp_loc", None))
        if not os.path.isdir(no_color_mask_dir):
            os.makedirs(no_color_mask_dir, exist_ok=True)
        no_color_mask_f_name = "{}/{}".format(no_color_mask_dir, file_name)

        pred_mask_np = pred_mask.cpu().numpy()

        if is_city_dataset:
            pred_mask_np = convert_to_cityscape_format(img=pred_mask_np)

        cv2.imwrite(no_color_mask_f_name, pred_mask_np)


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
    confmat = ConfusionMatrix()
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(val_loader)):
            input_img, target_label = batch['image'], batch['label']
            batch_size = input_img.shape[0]
            assert batch_size == 1, "We recommend to run segmentation evaluation with a batch size of 1"

            predict_and_save(
                opts=opts,
                input_tensor=input_img,
                file_name=batch["file_name"][0],
                orig_w=batch["im_width"][0].item(),
                orig_h=batch["im_height"][0].item(),
                model=model,
                target_label=target_label,
                device=device,
                mixed_precision_training=mixed_precision_training,
                confmat=confmat
            )

    acc_global, acc, iu = confmat.compute()
    logger.info("Quantitative results")
    print("global correct: {:.2f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.2f}".format(
        acc_global.item() * 100,
        ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100)
    )

    is_city_dataset = (getattr(opts, "dataset.name", "") == "cityscapes")
    if is_city_dataset:
        from .segmentation_utils.cityscapes_iou import eval_cityscapes
        pred_dir = "{}/predictions_no_cmap/".format(getattr(opts, "common.exp_loc", None))
        gt_dir = os.path.join(
            getattr(opts, "dataset.root_val", None),
            "gtFine/val/"
        )
        eval_cityscapes(pred_dir=pred_dir, gt_dir=gt_dir)


def predict_image(opts, image_fname, **kwargs):
    if not os.path.isfile(image_fname):
        logger.error("Image file does not exist at: {}".format(image_fname))

    orig_image = BaseImageDataset.read_image(path=image_fname)
    im_height, im_width = orig_image.shape[:2]

    res_h, res_w = tensor_size_from_opts(opts)
    input_img = cv2.resize(orig_image, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

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
            file_name=image_fname,
            orig_h=im_height,
            orig_w=im_width,
            model=model,
            target_label=None,
            device=device,
            mixed_precision_training=mixed_precision_training,
            orig_image=orig_image
        )


def predict_images_in_folder(opts, **kwargs):
    img_folder_path = getattr(opts, "evaluation.segmentation.path", None)
    if img_folder_path is None:
        logger.error(
            "Image folder is not passed. Please use --evaluation.segmentation.path as an argument to pass the location of image folder".format(
                img_folder_path))
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
        for image_fname in img_files:
            orig_img = BaseImageDataset.read_image(path=image_fname)
            im_height, im_width = orig_img.shape[:2]

            res_h, res_w = tensor_size_from_opts(opts)
            input_img = cv2.resize(orig_img, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

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
                file_name=image_fname,
                orig_h=im_height,
                orig_w=im_width,
                model=model,
                target_label=None,
                device=device,
                mixed_precision_training=mixed_precision_training,
                orig_image=orig_img
            )


def main_segmentation_evaluation(**kwargs):
    opts = get_segmentation_eval_arguments()

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
    # we disable the DDP setting for evaluating segmentation tasks
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

    eval_mode = getattr(opts, "evaluation.segmentation.mode", None)

    if eval_mode == "single_image":
        num_seg_classes = getattr(opts, "model.segmentation.n_classes", 21)
        assert num_seg_classes is not None

        # test a single image
        img_f_name = getattr(opts, "evaluation.segmentation.path", None)
        predict_image(opts, img_f_name, **kwargs)
    elif eval_mode == "image_folder":
        num_seg_classes = getattr(opts, "model.segmentation.n_classes", 21)
        assert num_seg_classes is not None

        # test all images in a folder
        # This is useful for generating results for test set
        predict_images_in_folder(opts=opts, **kwargs)
    elif eval_mode == "validation_set":
        # evaluate and compute stats for labeled image dataset
        # This is useful for generating results for validation set and compute quantitative results
        predict_labeled_dataset(opts=opts, **kwargs)
    else:
        logger.error(
            "Supported modes are single_image, image_folder, and validation_set. Got: {}".format(eval_mode)
        )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
import warnings
import torch
from torch import Tensor
from torch.utils import data
import cv2
from PIL import Image
from typing import Optional, Union, Dict
import argparse
import psutil
import time
import numpy as np
from torchvision.io import (
    read_image,
    read_file,
    decode_jpeg,
    ImageReadMode,
    decode_image,
)
import io

from utils import logger
from utils.ddp_utils import is_start_rank_node, is_master


class BaseImageDataset(data.Dataset):
    """
    Base Dataset class for Image datasets
    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ):
        if getattr(opts, "dataset.trove.enable", False):
            opts = self.load_from_server(opts=opts, is_training=is_training)

        root = (
            getattr(opts, "dataset.root_train", None)
            if is_training
            else getattr(opts, "dataset.root_val", None)
        )
        self.root = root
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.sampler_name = getattr(opts, "sampler.name", None)
        self.opts = opts

        image_device_cuda = getattr(self.opts, "dataset.decode_data_on_gpu", False)
        device = getattr(self.opts, "dev.device", torch.device("cpu"))
        use_cuda = False
        if image_device_cuda and (
            (isinstance(device, str) and device.find("cuda") > -1)
            or (isinstance(device, torch.device) and device.type.find("cuda") > -1)
        ):  # cuda could be cuda:0
            use_cuda = True

        if use_cuda and getattr(opts, "dataset.pin_memory", False):
            if is_master(opts):
                logger.error(
                    "For loading images on GPU, --dataset.pin-memory should be disabled."
                )

        self.device = device if use_cuda else torch.device("cpu")

        self.cached_data = (
            dict()
            if getattr(opts, "dataset.cache_images_on_ram", False) and is_training
            else None
        )
        if self.cached_data is not None:
            if not getattr(opts, "dataset.persistent_workers", False):
                if is_master(opts):
                    logger.error(
                        "For caching, --dataset.persistent-workers should be enabled."
                    )

        self.cache_limit = getattr(opts, "dataset.cache_limit", 80.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    @staticmethod
    def load_from_server(opts, is_training):
        try:
            from internal.utils.server_utils import load_from_data_server

            opts = load_from_data_server(opts=opts, is_training=is_training)
        except ImportError as e:
            import traceback
            traceback.print_exc()
            logger.error(
                "Unable to load data. Please load data manually. Error: {}".format(e)
            )

        return opts

    def _training_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def _validation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def _evaluation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def read_image_pil(self, path: str, *args, **kwargs):
        def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
            try:
                rgb_img = Image.open(inp_data).convert("RGB")
            except:
                rgb_img = None
            return rgb_img

        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                img_byte = self.cached_data[path]

            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
                    self.cached_data[path] = img_byte
            else:
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)  # in-memory data
            img = convert_to_rgb(img_byte)
        else:
            img = convert_to_rgb(path)
        return img

    def read_pil_image_torchvision(self, path: str):
        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                byte_img = self.cached_data[path]
            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                byte_img = read_file(path)
                self.cached_data[path] = byte_img
            else:
                byte_img = read_file(path)
        else:
            byte_img = read_file(path)
        img = decode_image(byte_img, mode=ImageReadMode.RGB)
        return img

    def read_image_tensor(self, path: str):
        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                byte_img = self.cached_data[path]
            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                byte_img = read_file(path)
                self.cached_data[path] = byte_img
            else:
                byte_img = read_file(path)
        else:
            byte_img = read_file(path)
        img = decode_jpeg(byte_img, device=self.device, mode=ImageReadMode.RGB)
        return img

    @staticmethod
    def read_mask_pil(path: str):
        try:
            mask = Image.open(path)
            if mask.mode != "L":
                logger.error("Mask mode should be L. Got: {}".format(mask.mode))
            return mask
        except:
            return None

    @staticmethod
    def read_image_opencv(path: str):
        warnings.warn(
            "The use of read_image_opencv function is depreciated. Please use read_image_pil",
            DeprecationWarning,
        )
        return cv2.imread(
            path, cv2.IMREAD_COLOR
        )  # Image is read in BGR Format and not RGB format

    @staticmethod
    def read_mask_opencv(path: str):
        warnings.warn(
            "The use of read_mask_opencv function is depreciated. Please use read_mask_pil",
            DeprecationWarning,
        )
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def convert_mask_to_tensor(mask):
        # convert to tensor
        mask = np.array(mask)
        if len(mask.shape) > 2 and mask.shape[-1] > 1:
            mask = np.ascontiguousarray(mask.transpose(2, 0, 1))
        return torch.as_tensor(mask, dtype=torch.long)

    @staticmethod
    def adjust_mask_value():
        return 0

    @staticmethod
    def class_names():
        pass

    def __repr__(self):
        return "{}(\n\troot={}\n\t is_training={})".format(
            self.__class__.__name__, self.root, self.is_training
        )

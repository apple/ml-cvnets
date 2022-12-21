#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import tarfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import torch

from utils import logger
from utils.download_utils import get_local_path

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T

IMAGENETv2_SPLIT_LINK_MAP = {
    "matched_frequency": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
        "extracted_folder_name": "imagenetv2-matched-frequency-format-val",
    },
    "threshold_0.7": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz",
        "extracted_folder_name": "imagenetv2-threshold0.7-format-val",
    },
    "top_images": {
        "url": "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz",
        "extracted_folder_name": "imagenetv2-top-images-format-val",
    },
}


@register_dataset(name="imagenet_v2", task="classification")
class Imagenetv2Dataset(BaseImageDataset):
    """
    `ImageNetv2 Dataset <https://arxiv.org/abs/1902.10811>`_ for studying the robustness of models trained on ImageNet dataset

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): ImageNetv2 should be used for evaluation only Default: False
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: True

    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = False,
        is_evaluation: Optional[bool] = True,
        *args,
        **kwargs,
    ) -> None:
        if is_training:
            logger.error(
                "{} can only be used for evaluation".format(self.__class__.__name__)
            )

        super().__init__(
            opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )

        split = getattr(opts, "dataset.imagenet_v2.split", None)
        if split is None or split not in IMAGENETv2_SPLIT_LINK_MAP.keys():
            logger.error(
                "Please specify split for ImageNetv2. Supported ImageNetv2 splits are: {}".format(
                    IMAGENETv2_SPLIT_LINK_MAP.keys()
                )
            )

        split_path = get_local_path(opts, path=IMAGENETv2_SPLIT_LINK_MAP[split]["url"])
        with tarfile.open(split_path) as tf:
            tf.extractall(self.root)

        root = Path(
            "{}/{}".format(
                self.root, IMAGENETv2_SPLIT_LINK_MAP[split]["extracted_folder_name"]
            )
        )
        file_names = list(root.glob("**/*.jpeg"))
        self.file_names = file_names

        setattr(opts, "dataset.collate_fn_name_train", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "imagenet_collate_fn")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.imagenet-v2.split",
            type=str,
            default="matched-frequency",
            help="ImageNetv2 dataset. Possible choices are: {}".format(
                [
                    f"{i + 1}: {split_name}"
                    for i, split_name in enumerate(IMAGENETv2_SPLIT_LINK_MAP.keys())
                ]
            ),
            choices=IMAGENETv2_SPLIT_LINK_MAP.keys(),
        )
        return parser

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Validation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        # same for validation and evaluation
        transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        # infer target label from the file name
        # file names are organized as SPLIT_NAME-format-val/class_idx/*.jpg
        # Example: All images in this folder (imagenetv2-matched-frequency-format-val/0/*.jpg) belong to class 0
        img_path = str(self.file_names[img_index])
        target = int(self.file_names[img_index].parent.name)

        input_img = self.read_image_pil(img_path)
        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=self.img_dtype
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["samples"] = data["image"]
        data["targets"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return len(self.file_names)

    def __repr__(self) -> str:
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tsamples={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            len(self.file_names),
            transforms_str,
        )

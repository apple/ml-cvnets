#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
from typing import Optional, List, Dict, Tuple
import numpy as np

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T


@register_dataset(name="ade20k", task="segmentation")
class ADE20KDataset(BaseImageDataset):
    """
    Dataset class for the ADE20K dataset

    The structure of the dataset should be something like this: ::

    ADEChallengeData2016/annotations/training/*.png
    ADEChallengeData2016/annotations/validation/*.png

    ADEChallengeData2016/images/training/*.jpg
    ADEChallengeData2016/images/validation/*.jpg

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    """

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        """

        :param opts: arguments
        :param is_training: Training or validation mode
        :param is_evaluation: Evaluation mode
        """
        super().__init__(
            opts=opts, is_training=is_training, is_evaluation=is_evaluation
        )
        root = self.root

        image_dir = os.path.join(
            root, "images", "training" if is_training else "validation"
        )
        annotation_dir = os.path.join(
            root, "annotations", "training" if is_training else "validation"
        )

        images = []
        masks = []
        for file_name in os.listdir(image_dir):
            if file_name.endswith(".jpg"):
                img_f_name = "{}/{}".format(image_dir, file_name)
                mask_f_name = "{}/{}".format(
                    annotation_dir, file_name.replace("jpg", "png")
                )

                if os.path.isfile(img_f_name) and os.path.isfile(mask_f_name):
                    images.append(img_f_name)
                    masks.append(mask_f_name)

        self.images = images
        self.masks = masks
        self.ignore_label = 255
        self.bgrnd_idx = 0
        setattr(
            opts, "model.segmentation.n_classes", len(self.class_names()) - 1
        )  # ignore background

        # set the collate functions for the dataset
        # For evaluation, we use PyTorch's default collate function. So, we set to collate_fn_name_eval to None
        setattr(opts, "dataset.collate_fn_name_train", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", None)

    def _training_transforms(self, size: tuple):
        first_aug = T.RandomShortSizeResize(opts=self.opts)
        aug_list = [
            T.RandomHorizontalFlip(opts=self.opts),
            T.RandomCrop(opts=self.opts, size=size, ignore_idx=self.ignore_label),
        ]

        if getattr(self.opts, "image_augmentation.random_gaussian_noise.enable", False):
            aug_list.append(T.RandomGaussianBlur(opts=self.opts))

        if getattr(self.opts, "image_augmentation.photo_metric_distort.enable", False):
            aug_list.append(T.PhotometricDistort(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_rotate.enable", False):
            aug_list.append(T.RandomRotate(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_order.enable", False):
            new_aug_list = [
                first_aug,
                T.RandomOrder(opts=self.opts, img_transforms=aug_list),
                T.ToTensor(opts=self.opts),
            ]
            return T.Compose(opts=self.opts, img_transforms=new_aug_list)
        else:
            aug_list.insert(0, first_aug)
            aug_list.append(T.ToTensor(opts=self.opts))
            return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = [T.Resize(opts=self.opts), T.ToTensor(opts=self.opts)]
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: tuple, *args, **kwargs):
        aug_list = []
        if getattr(self.opts, "evaluation.segmentation.resize_input_images", False):
            # we want to resize while maintaining aspect ratio. So, we pass img_size argument to resize function
            aug_list.append(T.Resize(opts=self.opts, img_size=min(size)))

        aug_list.append(T.ToTensor(opts=self.opts))
        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def __getitem__(self, batch_indexes_tup: Tuple[int, int, int]) -> Dict:
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        crop_size = (crop_size_h, crop_size_w)

        if self.is_training:
            _transform = self._training_transforms(size=crop_size)
        elif self.is_evaluation:
            _transform = self._evaluation_transforms(size=crop_size)
        else:
            _transform = self._validation_transforms(size=crop_size)

        mask = self.read_mask_pil(self.masks[img_index])
        img = self.read_image_pil(self.images[img_index])

        if (img.size[0] != mask.size[0]) or (img.size[1] != mask.size[1]):
            logger.error(
                "Input image and mask sizes are different. Input size: {} and Mask size: {}".format(
                    img.size, mask.size
                )
            )

        data = {"image": img}
        if not self.is_evaluation:
            data["mask"] = mask

        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = self.convert_mask_to_tensor(mask)

        data["label"] = data["mask"] - 1  # ignore background during training
        del data["mask"]

        if self.is_evaluation:
            im_width, im_height = img.size
            img_name = self.images[img_index].split(os.sep)[-1].replace("jpg", "png")
            data["file_name"] = img_name
            data["im_width"] = im_width
            data["im_height"] = im_height

        return data

    @staticmethod
    def adjust_mask_value():
        return 1

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def color_palette() -> List:
        color_codes = [
            [0, 0, 0],  # background
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]
        color_codes = np.asarray(color_codes).flatten()
        return list(color_codes)

    @staticmethod
    def class_names() -> List:
        return [
            "background",
            "wall",
            "building",
            "sky",
            "floor",
            "tree",
            "ceiling",
            "road",
            "bed ",
            "windowpane",
            "grass",
            "cabinet",
            "sidewalk",
            "person",
            "earth",
            "door",
            "table",
            "mountain",
            "plant",
            "curtain",
            "chair",
            "car",
            "water",
            "painting",
            "sofa",
            "shelf",
            "house",
            "sea",
            "mirror",
            "rug",
            "field",
            "armchair",
            "seat",
            "fence",
            "desk",
            "rock",
            "wardrobe",
            "lamp",
            "bathtub",
            "railing",
            "cushion",
            "base",
            "box",
            "column",
            "signboard",
            "chest of drawers",
            "counter",
            "sand",
            "sink",
            "skyscraper",
            "fireplace",
            "refrigerator",
            "grandstand",
            "path",
            "stairs",
            "runway",
            "case",
            "pool table",
            "pillow",
            "screen door",
            "stairway",
            "river",
            "bridge",
            "bookcase",
            "blind",
            "coffee table",
            "toilet",
            "flower",
            "book",
            "hill",
            "bench",
            "countertop",
            "stove",
            "palm",
            "kitchen island",
            "computer",
            "swivel chair",
            "boat",
            "bar",
            "arcade machine",
            "hovel",
            "bus",
            "towel",
            "light",
            "truck",
            "tower",
            "chandelier",
            "awning",
            "streetlight",
            "booth",
            "television receiver",
            "airplane",
            "dirt track",
            "apparel",
            "pole",
            "land",
            "bannister",
            "escalator",
            "ottoman",
            "bottle",
            "buffet",
            "poster",
            "stage",
            "van",
            "ship",
            "fountain",
            "conveyer belt",
            "canopy",
            "washer",
            "plaything",
            "swimming pool",
            "stool",
            "barrel",
            "basket",
            "waterfall",
            "tent",
            "bag",
            "minibike",
            "cradle",
            "oven",
            "ball",
            "food",
            "step",
            "tank",
            "trade name",
            "microwave",
            "pot",
            "animal",
            "bicycle",
            "lake",
            "dishwasher",
            "screen",
            "blanket",
            "sculpture",
            "hood",
            "sconce",
            "vase",
            "traffic light",
            "tray",
            "ashcan",
            "fan",
            "pier",
            "crt screen",
            "plate",
            "monitor",
            "bulletin board",
            "shower",
            "radiator",
            "glass",
            "clock",
            "flag",
        ]

    def __repr__(self) -> str:
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        elif self.is_evaluation:
            transforms_str = self._evaluation_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return (
            "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\ttransforms={}\n)".format(
                self.__class__.__name__,
                self.root,
                self.is_training,
                len(self.images),
                transforms_str,
            )
        )

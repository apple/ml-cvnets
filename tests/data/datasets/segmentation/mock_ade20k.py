#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Optional

from PIL import Image

from data.datasets.segmentation.ade20k import ADE20KDataset

TOTAL_SAMPLES = 100


class MockADE20KDataset(ADE20KDataset):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset

        Specifically, we replace the samples and targets with random data so that actual dataset is not
        required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.images = ["dummy_img_path.jpg" for _ in range(TOTAL_SAMPLES)]
        self.masks = ["dummy_mask_path.png" for _ in range(TOTAL_SAMPLES)]
        self.ignore_label = 255
        self.background_idx = 0
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.check_dataset()

    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function

        Instead of reading a PIL RGB image at location specified by `path`, a PIL
        RGB image of size (20, 40) returned.
        """
        return Image.new("RGB", (20, 30))

    @staticmethod
    def read_mask_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_mask_pil function

        Instead of reading a mask at location specified by `path`, a PIL mask image of
        size (20, 40) is returned.
        """
        return Image.new("L", (20, 30))

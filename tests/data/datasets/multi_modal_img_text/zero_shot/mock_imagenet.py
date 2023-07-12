#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
import random
from typing import Optional

from PIL import Image

from data.datasets.multi_modal_img_text.zero_shot.imagenet import (
    ImageNetDatasetZeroShot,
    generate_text_prompts_clip,
)

TOTAL_SAMPLES = 100


class MockImageNetDatasetZeroShot(ImageNetDatasetZeroShot):
    """Mock the ImageNetDatasetZeroShot without initializing from image folders."""

    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset.

        Specifically, we replace the samples and targets with random data so that actual
        dataset is not required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.samples = [
            ["img_path", random.randint(1, 4)] for _ in range(TOTAL_SAMPLES)
        ]
        self.text_prompts = [
            generate_text_prompts_clip(class_name) for class_name in ["cat", "dog"]
        ]
        self.targets = [class_id for img_path, class_id in self.samples]
        self.imgs = [img_path for img_path, class_id in self.samples]
        self.is_training = is_training
        self.is_evaluation = is_evaluation

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
import random
from typing import Optional

from PIL import Image

from data.datasets.classification.imagenet import ImageNetDataset
from data.datasets.classification.imagenet_a import ImageNetADataset
from data.datasets.classification.imagenet_r import ImageNetRDataset
from data.datasets.classification.imagenet_sketch import ImageNetSketchDataset

TOTAL_SAMPLES = 100


class MockImageNetDataset(ImageNetDataset):
    def __init__(
        self,
        opts: argparse.Namespace,
        is_training: bool = False,
        is_evaluation: bool = False,
        *args,
        **kwargs
    ) -> None:
        """Mock the init logic for ImageNet dataset.

        Specifically, we replace the samples and targets with random data so that actual dataset is not
        required for testing purposes.
        """
        # super() is not called here intentionally.
        self.opts = opts
        self.root = None
        self.samples = [
            ["img_path", random.randint(1, 4)] for _ in range(TOTAL_SAMPLES)
        ]
        self.targets = [class_id for img_path, class_id in self.samples]
        self.imgs = [img_path for img_path, class_id in self.samples]
        self.is_training = is_training
        self.is_evaluation = is_evaluation

    @staticmethod
    def read_image_pil(path: str) -> Optional[Image.Image]:
        """Mock the init logic for read_image_pil function.

        Instead of reading a PIL image at location specified by `path`, a random PIL
        image is returned. The randomness in height and width dimensions may allow us to
        catch errors in transform functions.
        """
        return Image.new("RGB", (random.randint(10, 20), random.randint(10, 20)))


class MockImageNetADataset(MockImageNetDataset, ImageNetADataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetA dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.n_classes = 1000
        self.post_init_checks()


class MockImageNetRDataset(MockImageNetDataset, ImageNetRDataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetR dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.n_classes = 1000
        self.post_init_checks()


class MockImageNetSketchDataset(MockImageNetDataset, ImageNetSketchDataset):
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        """Mock the init logit for ImageNetSketch dataset."""
        MockImageNetDataset.__init__(self, opts, *args, **kwargs)
        self.n_classes = 1000
        self.post_init_checks()

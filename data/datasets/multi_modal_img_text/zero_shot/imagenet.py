#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List

from torchvision.datasets import ImageFolder

from data.datasets.multi_modal_img_text.zero_shot import (
    ZERO_SHOT_DATASET_REGISTRY,
    BaseZeroShotDataset,
)
from data.datasets.multi_modal_img_text.zero_shot.imagenet_class_names import (
    IMAGENET_CLASS_NAMES,
)
from data.datasets.multi_modal_img_text.zero_shot.templates import (
    generate_text_prompts_clip,
)


@ZERO_SHOT_DATASET_REGISTRY.register(name="imagenet")
class ImageNetDatasetZeroShot(BaseZeroShotDataset, ImageFolder):
    """ImageNet Dataset for zero-shot evaluation of Image-text models.

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        BaseZeroShotDataset.__init__(self, opts=opts, *args, **kwargs)
        root = self.root
        ImageFolder.__init__(
            self, root=root, transform=None, target_transform=None, is_valid_file=None
        )

        # TODO: Refactor BaseZeroShotDataset to inherit from
        # BaseImageClassificationDataset then inherit from ImageNetDataset instead of
        # ImageFolder. Rename the base class to BaseZeroShotClassificationDataset.
        assert len(list(self.class_to_idx.keys())) == len(self.class_names()), (
            "Number of classes from ImageFolder do not match the number of ImageNet"
            " classes."
        )

    @classmethod
    def class_names(cls) -> List[str]:
        """Return the name of the classes present in the dataset."""
        return IMAGENET_CLASS_NAMES

    @staticmethod
    def generate_text_prompts(class_name: str) -> List[str]:
        """Return a list of prompts for the given class name."""
        return generate_text_prompts_clip(class_name)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return super(ImageFolder, self).__len__()

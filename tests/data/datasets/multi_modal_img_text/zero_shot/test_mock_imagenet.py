#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Type

from tests.configs import get_config
from tests.data.datasets.multi_modal_img_text.zero_shot.mock_imagenet import (
    MockImageNetDatasetZeroShot,
)


def test_imagenet_dataset_zero_shot(
    config_file_path: str = "tests/data/datasets/multi_modal_img_text/zero_shot/dummy_imagenet_config.yaml",
    mock_dataset_class: Type[MockImageNetDatasetZeroShot] = MockImageNetDatasetZeroShot,
) -> None:
    """Test for ImageNet zero-shot.

    Similar test to ImageNet test but only for validation because zero-shot datasets are
    not supposed to be used for training. We also test the text prompts.
    """
    opts = get_config(config_file=config_file_path)

    imagenet_zero_shot_dataset = mock_dataset_class(
        opts, is_training=False, is_evaluation=False
    )

    for image_id in range(2):
        data = imagenet_zero_shot_dataset[image_id]
        # values from config file
        assert len(data) == 3, "ImageNet zero shot should return a tuple of 3."
        img_path, text_prompts, target = data
        assert isinstance(img_path, str), "ImageNet zero shot should return (str, ...)."
        assert (
            isinstance(text_prompts, list)
            and isinstance(text_prompts[0], list)
            and isinstance(text_prompts[0][0], str)
        ), "ImageNet zero shot should return (..., list[list[str]], ...)."
        assert isinstance(target, int), "ImageNet zero shot should return (..., int)."

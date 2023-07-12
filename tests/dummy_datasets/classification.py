#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Dict, Tuple

import torch
import torch.utils.data as data


class DummyClassificationDataset(data.Dataset):
    """
    Dummy Classification Dataset for CI/CD testing

    Args:
        opts: command-line arguments

    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.n_classes = 1000
        setattr(opts, "model.classification.n_classes", self.n_classes)
        setattr(
            opts,
            "dataset.collate_fn_name_train",
            "image_classification_data_collate_fn",
        )
        setattr(
            opts, "dataset.collate_fn_name_val", "image_classification_data_collate_fn"
        )
        setattr(
            opts, "dataset.collate_fn_name_test", "image_classification_data_collate_fn"
        )

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        input_img = torch.randn(size=(3, crop_size_h, crop_size_w), dtype=torch.float)
        target = torch.randint(low=0, high=self.n_classes, size=(1,)).long()

        return {
            "samples": input_img,
            "targets": target,
            "sample_id": torch.randint(low=0, high=1000, size=(1,)).long(),
        }

    def __len__(self) -> int:
        return 10

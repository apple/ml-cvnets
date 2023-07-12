#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Dict, Tuple

import torch
import torch.utils.data as data


class DummySegmentationDataset(data.Dataset):
    """
    Dummy Segmentation Dataset for CI/CD testing

    Args:
        opts: command-line arguments

    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.n_classes = 20
        setattr(opts, "model.segmentation.n_classes", self.n_classes)
        setattr(opts, "dataset.collate_fn_name_train", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_test", None)

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        input_img = torch.randn(size=(3, crop_size_h, crop_size_w), dtype=torch.float)
        target = torch.randint(
            low=0, high=self.n_classes, size=(crop_size_h, crop_size_w)
        ).long()

        return {"samples": input_img, "targets": target}

    def __len__(self) -> int:
        return 10

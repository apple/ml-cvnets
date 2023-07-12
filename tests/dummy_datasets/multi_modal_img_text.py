#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Dict, Tuple

import torch
import torch.utils.data as data


class DummyMultiModalImageTextDataset(data.Dataset):
    """
    Dummy Dataset for CI/CD testing

    Args:
        opts: command-line arguments

    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.context_length = 5
        self.vocab_size = 100
        setattr(opts, "dataset.text_vocab_size", self.vocab_size)
        setattr(opts, "dataset.text_context_length", self.context_length)

        setattr(
            opts, "dataset.collate_fn_name_train", "multi_modal_img_text_collate_fn"
        )
        setattr(opts, "dataset.collate_fn_name_val", "multi_modal_img_text_collate_fn")
        setattr(opts, "dataset.collate_fn_name_test", "multi_modal_img_text_collate_fn")

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup

        input_img = torch.randn(size=(3, crop_size_h, crop_size_w), dtype=torch.float)
        text = torch.randint(
            low=0, high=self.vocab_size, size=(self.context_length,), dtype=torch.int
        )

        return {
            "samples": {"image": input_img, "text": text, "padding_mask": None},
            "targets": torch.randint(low=0, high=1, size=(1,)).long(),
        }

    def __len__(self) -> int:
        return 10

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Tuple, Dict
import torch
import torch.utils.data as data


class DummyVideoClassificationDataset(data.Dataset):
    """
    Dummy Video Classification Dataset for CI/CD testing

    Args:
        opts: command-line arguments

    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.n_classes = 100
        self.frame_stack_format = getattr(
            opts, "video_reader.frame_stack_format", "sequence_first"
        )
        self.stack_frame_dim = 1 if self.frame_stack_format == "channel_first" else 0

        setattr(opts, "model.video_classification.n_classes", self.n_classes)
        setattr(opts, "dataset.collate_fn_name_train", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "default_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "default_collate_fn")

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        (
            crop_size_h,
            crop_size_w,
            img_index,
            n_frames_to_sample,
            clips_per_video,
        ) = batch_indexes_tup

        # for CI/CD, we do not use clips as they are merged with batch size (by default) inside collate function
        clips_per_video = 1

        if self.frame_stack_format == "channel_first":
            input_img = torch.randn(
                size=(3, n_frames_to_sample, crop_size_h, crop_size_w),
                dtype=torch.float,
            )
        else:
            input_img = torch.randn(
                size=(n_frames_to_sample, 3, crop_size_h, crop_size_w),
                dtype=torch.float,
            )
        random_label = torch.randint(low=0, high=self.n_classes, size=(1,)).item()
        # create a 0-dim target
        target = torch.tensor(random_label).long()

        return {
            "samples": input_img,
            "targets": target,
            "sample_id": torch.randint(
                low=0, high=1000, size=(clips_per_video,)
            ).long(),
        }

    def __len__(self) -> int:
        return 10

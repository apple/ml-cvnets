#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Tuple

from options.utils import extract_opts_with_prefix_replacement
from utils import logger


class BaseZeroShotDataset(object):
    """Base Dataset class for zero shot tasks.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        # we need to set the default value of this one
        if getattr(opts, "dataset.multi_modal_img_text.zero_shot.trove.enable", False):
            try:
                from internal.utils.server_utils import load_from_data_server

                opts = load_from_data_server(
                    opts=opts,
                    is_training=False,
                    is_evaluation=False,  # use root_val / dir_val
                    arg_prefix="dataset.multi_modal_img_text.zero_shot",
                )
            except Exception as e:
                logger.error("Unable to load from the server. Error: {}".format(str(e)))

        # Extracting zero-shot options to be able to build them separately in
        # child classes
        dataset_opts = extract_opts_with_prefix_replacement(
            opts,
            match_prefix="dataset.multi_modal_img_text.zero_shot.",
            replacement_prefix="dataset.",
        )
        dataset_opts = vars(dataset_opts)
        dataset_opts.update(
            {
                "dataset.num_samples_per_category": -1,
                "dataset.percentage_of_samples": 100.0,
            }
        )
        self.dataset_opts = argparse.Namespace(**dataset_opts)

        root = getattr(opts, "dataset.multi_modal_img_text.zero_shot.root_val")
        self.root = root
        self.opts = opts

        # Initialize text prompts using the static method `class_names`.
        text_prompts = []
        for class_id, class_name in enumerate(self.class_names()):
            text_prompts.append(self.generate_text_prompts(class_name.lower()))
        self.text_prompts = text_prompts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseZeroShotDataset:
            # Don't re-register arguments in subclasses that don't override
            # `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--dataset.multi-modal-img-text.zero-shot-eval",
            action="store_true",
            default=False,
            help="Use zero shot evaluation. Defaults to False.",
        )
        group.add_argument(
            "--dataset.multi-modal-img-text.zero-shot.name",
            type=str,
            default=None,
            help="Name of the dataset for zero-shot evaluation. Defaults to None.",
        )
        group.add_argument(
            "--dataset.multi-modal-img-text.zero-shot.root-val",
            type=str,
            default=None,
            help="Location of the dataset for zero-shot evaluation. Defaults to None.",
        )
        return parser

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError(
            "Sub-classes should implement `__len__` that returns the number of samples"
            " in the dataset."
        )

    def __getitem__(self, img_index: int) -> Tuple[str, List[List[str]], int]:
        """Return image path and text templates for a given image index.

        Args:
            img_index: Index of the image.

        Returns:
            Tuple containing image path, list of captions, and image label
        """
        img_path, target = self.samples[img_index]
        return img_path, self.text_prompts, target

    @classmethod
    def class_names(cls) -> List[str]:
        """Return the name of the classes in the dataset. Label is index in the list.

        The order of class names in the returned list determine the numerical class
        label.
        """
        raise NotImplementedError(
            "Sub-classes should define `class_names` that returns the list of class"
            " names in the order of class labels."
        )

    @staticmethod
    def generate_text_prompts(class_name: str) -> List[str]:
        """Return a list of prompts for the given class name."""
        raise NotImplementedError(
            "Sub-classes should define `generate_text_prompts` that creates a list of"
            " prompts for a given class name."
        )

    def __repr__(self) -> str:
        return "{}(root={})".format(self.__class__.__name__, self.root)

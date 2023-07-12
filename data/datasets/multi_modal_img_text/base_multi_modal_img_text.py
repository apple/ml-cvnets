#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from PIL import Image
from torch import Tensor

from data.collate_fns import COLLATE_FN_REGISTRY
from data.datasets.dataset_base import BaseImageDataset
from data.datasets.multi_modal_img_text.zero_shot import (
    BaseZeroShotDataset,
    build_zero_shot_dataset,
)
from data.datasets.utils.text import caption_preprocessing
from data.text_tokenizer import build_tokenizer
from data.transforms import image_pil as T
from data.transforms.common import Compose
from utils import logger
from utils.ddp_utils import is_master, is_start_rank_node


class BaseMultiModalImgText(BaseImageDataset):
    """
    Base class for Image-Text multi-modal learning

    Args:
        opts: command-line arguments
    """

    __SEPARATOR = ":"

    def __init__(
        self,
        opts,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(
            opts=opts,
            *args,
            **kwargs,
        )

        self.is_master_node = is_master(opts)
        self.is_start_rank_node = is_start_rank_node(opts)

        self.text_tokenizer = build_tokenizer(opts=opts, *args, **kwargs)
        self.context_length = getattr(
            opts, "dataset.multi_modal_img_text.context_length"
        )

        # for sharing padding index across the entire cvnets framework, we will
        # use a special variable "dataset.padding_index".
        setattr(opts, "dataset.padding_index", None)
        self.padding_index = getattr(opts, "dataset.padding_index")

        vocab_size = self.text_tokenizer.get_vocab_size()
        if vocab_size is None or vocab_size == -1:
            logger.error(
                "Vocab size can't be None or -1 in {}. Got: {}".format(
                    self.__class__.__name__, vocab_size
                )
            )
        self.vocab_size = vocab_size
        setattr(opts, "dataset.text_vocab_size", vocab_size)
        setattr(opts, "dataset.text_context_length", self.context_length)

        setattr(
            opts,
            "dataset.collate_fn_name_train",
            "multi_modal_img_text_collate_fn",
        )
        setattr(
            opts,
            "dataset.collate_fn_name_val",
            "multi_modal_img_text_collate_fn",
        )
        setattr(
            opts,
            "dataset.collate_fn_name_test",
            "multi_modal_img_text_collate_fn",
        )

        self.zero_shot_dataset = self.get_zero_shot_dataset(*args, **kwargs)
        self.cached_zero_shot_captions = None

        self.cache_loc = os.path.join(self.root, ".img_text_tar_cache")
        if self.is_training:
            # Folder where we will download data
            # TODO: Training data can't fit on a single node, so we save/cache subset of data on each node.
            # In future, we may enable caching for validation data.
            try:
                os.makedirs(self.cache_loc, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create cache location directory: {e}")

        self.dataset = self.get_dataset(*args, **kwargs)

    def get_zero_shot_dataset(self, *args, **kwargs) -> Optional[BaseZeroShotDataset]:
        """If zero-shot evaluation is enabled, zero-shot dataset is returned.
        Otherwise, None is returned
        """
        zero_shot_eval = (
            False
            if self.is_training
            else getattr(
                self.opts, "dataset.multi_modal_img_text.zero_shot_eval", False
            )
        )
        if zero_shot_eval:
            zero_shot_dataset = build_zero_shot_dataset(opts=self.opts, *args, **kwargs)
        else:
            zero_shot_dataset = None
        return zero_shot_dataset

    def get_dataset(self, *args, **kwargs) -> Any:
        """Helper function to get the dataset. Child classes must override this function"""
        raise NotImplementedError

    def share_dataset_arguments(self) -> Dict[str, Any]:
        """Returns the number of classes in detection dataset along with super-class arguments."""
        share_dataset_specific_opts: Dict[str, Any] = super().share_dataset_arguments()
        share_dataset_specific_opts["dataset.text_vocab_size"] = self.vocab_size
        share_dataset_specific_opts["dataset.text_context_length"] = self.context_length
        return share_dataset_specific_opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""

        if cls != BaseMultiModalImgText:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--dataset.multi-modal-img-text.context-length",
            type=int,
            default=77,
            help="Context length for the text model. Defaults to 77, the same as in CLIP paper.",
        )

        group.add_argument(
            "--dataset.multi-modal-img-text.trunc-seq-len",
            action="store_true",
            default=False,
            help="Many sequences in a batch do not have lengths equal to specified context length. Enabling this flag "
            "allows us to truncate the sequences such that the sequence length of a batch is equal to sequence "
            "with max. non-padded tokens. Defaults to False.",
        )

        return parser

    def _transform_text(self, text_tensor: Tensor) -> Tuple[Tensor, int]:
        """Helper function to transform the text tensor. If the text tensor is smaller
        than the context length, it pads it and replaces the last token with EOT token.

        Args:
            text_tensor: Text tensor with N tokens. Shape is (N,).

        Returns:
            A Tuple of text tensor (whole length is equal to context length) and length of the tensor.
        """
        captions_tensor = torch.zeros(size=(self.context_length,), dtype=torch.long)

        text_len = text_tensor.shape[0]
        if text_len > self.context_length:
            text_tensor = text_tensor[: self.context_length]
            text_tensor[-1] = self.text_tokenizer.get_eot_token()
            text_len = self.context_length
        captions_tensor[:text_len] = text_tensor[:text_len]
        return captions_tensor, text_len

    def _training_transforms(
        self, size: Tuple[int, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data augmentation during training.

        The default order is RandomResizedCrop, Optional[RandAugment or AutoAugment], ToTensor, Optional[RandomErase]

        Args:
            size: Size for resizing the input image. Expected to be a tuple (height, width)

        Returns:
            An instance of `data.transforms.image_pil.BaseTransformation.`

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
        ]
        auto_augment = getattr(
            self.opts, "image_augmentation.auto_augment.enable", False
        )
        rand_augment = getattr(
            self.opts, "image_augmentation.rand_augment.enable", False
        )
        if auto_augment and rand_augment:
            logger.error(
                "AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"
            )
        elif auto_augment:
            aug_list.append(T.AutoAugment(opts=self.opts))
        elif rand_augment:
            aug_list.append(T.RandAugment(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable", False):
            aug_list.append(T.RandomErasing(opts=self.opts))

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(
        self, size: Union[Tuple, int], *args, **kwargs
    ) -> T.BaseTransformation:
        """Data transforms during validation or evaluation
         The order is Resize, CenterCrop, ToTensor

         Args:
            size: Size for resizing the input image. Expected to be an integer (width=height) or a tuple (height, width)

        Returns:
            An instance of `data.transforms.image_pil.BaseTransformation.`
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return Compose(opts=self.opts, img_transforms=aug_list)

    def _process_img_caption(
        self,
        input_img: Image.Image,
        captions_str: Union[str, List[str], List[List[str]]],
        img_transform_fn: T.BaseTransformation,
        zero_shot: bool,
    ) -> Tuple[Tensor, Tensor, int]:
        """Apply data augmentation to images and pre-processing to text captions

        Args:
            input_img: Input PIL Image
            captions_str: Text captions
            img_transform_fn: Image transform functions
            zero_shot: zero shot evaluation or not

        Returns:
            A tuple of image tensor, caption tensor, and max. sequence length of a sequence in caption tensor
        """

        data = {"image": input_img}
        img_tensor = img_transform_fn(data)["image"]

        if zero_shot and self.cached_zero_shot_captions is not None:
            return (
                img_tensor,
                self.cached_zero_shot_captions[0],
                self.cached_zero_shot_captions[1],
            )

        max_seq_len = 0
        # process caption
        if isinstance(captions_str, str):
            captions_tensor, max_seq_len = self._transform_text(
                self.text_tokenizer(caption_preprocessing(captions_str))
            )
        elif isinstance(captions_str, List):
            captions_tensor = []
            for captions_str_i in captions_str:
                if isinstance(captions_str_i, List):
                    # captions_str is [ [Num_templates_per_class] * Num_classes]
                    captions_tensor_i = []
                    for (
                        captions_str_i_j
                    ) in captions_str_i:  # number of templates per class
                        seq, seq_len = self._transform_text(
                            self.text_tokenizer(caption_preprocessing(captions_str_i_j))
                        )
                        captions_tensor_i.append(seq)
                        max_seq_len = max(max_seq_len, seq_len)
                    captions_tensor_i = torch.stack(captions_tensor_i, dim=0)
                    captions_tensor.append(captions_tensor_i)
                elif isinstance(captions_str_i, str):
                    # captions_str is [Num_templates_per_image]
                    seq, seq_len = self._transform_text(
                        self.text_tokenizer(caption_preprocessing(captions_str_i))
                    )
                    captions_tensor.append(seq)
                    max_seq_len = max(max_seq_len, seq_len)
                else:
                    logger.error(
                        "Got captions_str of type {}: {} from {}".format(
                            type(captions_str_i), captions_str_i, captions_str
                        )
                    )
            # the shape of tensor is [Num_classes, captions_per_class, caption_length]
            # or [Captions_per_image, caption_length]
            captions_tensor = torch.stack(captions_tensor, dim=0)
        else:
            captions_tensor = None
            logger.error(
                "Captions should be either string, List[String] or List[List[str]]"
            )

        if zero_shot and self.cached_zero_shot_captions is None:
            self.cached_zero_shot_captions = (captions_tensor, max_seq_len)

        return img_tensor, captions_tensor, max_seq_len

    def get_zero_shot_pair(
        self, img_index: int
    ) -> Tuple[Image.Image, Union[str, List[str], List[List[str]]], int]:
        """Get image-text pair for zero-shot dataset along with classification label.

        Args:
            img_index: Image index

        Returns:
            A tuple of PIL image, captions, and class label
        """
        img_path, captions_str, class_label = self.zero_shot_dataset[img_index]
        input_img = self.read_image_pil(img_path)
        return input_img, captions_str, class_label

    def get_dataset_pair(self, img_index: int) -> Any:
        """Get image-text pair from the dataset. Sub-classes must implement this method."""
        raise NotImplementedError

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and label of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["samples"]["text"]: Shape is
                * [Context_Length] (single caption, as in CLIP datasets)
                * [Num_classes, Num_Captions, Context_length] (multiple captions per class, as in 0-shot Imagenet dataset)
            output_data["samples"]["padding_mask"]: Same as output_data["samples"]["text"]
            output_data["samples"]["max_seq_len"]: Shape is [1]
            output_data["targets"]: Shape is [1]
        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index
        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        if self.zero_shot_dataset is not None:
            # read captions and image path from conceptual captions dataset
            # read captions and image path from zero-shot dataset
            input_img, captions_str, class_label = self.get_zero_shot_pair(
                img_index=img_index
            )
        else:
            input_img, captions_str, class_label = self.get_dataset_pair(
                img_index=img_index
            )

        if input_img is None:
            captions_tensor = torch.zeros(size=(self.context_length,), dtype=torch.long)
            data = {
                "samples": {
                    "image": torch.zeros(size=(3, crop_size_h, crop_size_w)),
                    "text": captions_tensor,
                    "padding_mask": (captions_tensor == self.padding_index)
                    if self.padding_index is not None
                    else None,
                    "max_seq_len": self.context_length,
                },
                "targets": -1,
            }
        else:
            (img_tensor, captions_tensor, max_seq_len,) = self._process_img_caption(
                input_img=input_img,
                captions_str=captions_str,
                img_transform_fn=transform_fn,
                zero_shot=self.zero_shot_dataset is not None,
            )

            data = {
                "samples": {
                    "image": img_tensor,
                    "text": captions_tensor,
                    "padding_mask": (captions_tensor == self.padding_index)
                    if self.padding_index is not None
                    else None,
                    "max_seq_len": max_seq_len,
                },
                "targets": class_label,
            }

        if self.zero_shot_dataset is not None:
            data["zero_shot"] = 1

        return data

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += f"\n\t zero_shot={self.zero_shot_dataset}"
        return extra_repr_str


@COLLATE_FN_REGISTRY.register(name="multi_modal_img_text_collate_fn")
def multi_modal_img_text_collate_fn(
    batch: List[Mapping[str, Union[Tensor, Mapping[str, Tensor]]]],
    opts: argparse.Namespace,
) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields."""
    images = []
    text_tokens = []
    padding_mask = []
    labels = []

    truncate_seq_len = getattr(opts, "dataset.multi_modal_img_text.trunc_seq_len")

    zero_shot = batch[0].pop("zero_shot", 0)

    max_seq_len_in_batch = 1  # at least one token is required in the sequence
    for i, batch_i in enumerate(batch):
        inputs_i = batch_i.pop("samples")
        img_tensor = inputs_i.pop("image", None)
        if img_tensor is None:
            continue
        images.append(img_tensor)
        labels.append(batch_i.pop("targets"))

        text_data = inputs_i.pop("text")
        pad_mask = inputs_i.pop("padding_mask", None)
        max_seq_len_in_batch = max(max_seq_len_in_batch, inputs_i.pop("max_seq_len", 0))
        if not zero_shot or (zero_shot and i == 0):
            # For zero-shot, all text captions are the same
            # so, we only aggregate for one batch element
            text_tokens.append(text_data)
            if pad_mask is not None:
                padding_mask.append(pad_mask)

    images = torch.stack(images, dim=0)
    text_tokens = torch.stack(text_tokens, dim=0)

    # truncate tokens based on the max. seq length
    if not truncate_seq_len:
        max_seq_len_in_batch = text_tokens.shape[-1]
    text_tokens = text_tokens[..., :max_seq_len_in_batch]

    if len(padding_mask) != 0:
        padding_mask = torch.stack(padding_mask, dim=0)
        padding_mask = padding_mask[..., :max_seq_len_in_batch]
    else:
        padding_mask = None

    labels = torch.tensor(labels, dtype=torch.long)

    channels_last = getattr(opts, "common.channels_last")
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {
        "samples": {
            "image": images,
            "text": text_tokens,
            "padding_mask": padding_mask,
        },
        "targets": labels,
    }

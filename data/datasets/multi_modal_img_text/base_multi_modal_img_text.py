#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple, Dict, List, Union
import torch
from torch import Tensor
import argparse
import ftfy
import re
import urllib
import os

from utils import logger
from utils.ddp_utils import is_master, is_start_rank_node

from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T
from ...collate_fns import register_collate_fn
from ...text_tokenizer import build_tokenizer
from .zero_shot import build_zero_shot_dataset


class BaseMultiModalImgText(BaseImageDataset):
    """
    Base class for Image-Text multi-modal learning

    Args:
        opts: command-line arguments
        is_training (Optional[bool]): A flag used to indicate training or validation mode. Default: True
        is_evaluation (Optional[bool]): A flag used to indicate evaluation (or inference) mode. Default: False

    """

    __separator = ":"

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:

        super().__init__(
            opts=opts,
            is_training=is_training,
            is_evaluation=is_evaluation,
            *args,
            **kwargs
        )

        self.is_master_node = is_master(opts)
        self.is_start_rank_node = is_start_rank_node(opts)

        self.text_tokenizer = build_tokenizer(opts=opts, *args, **kwargs)
        # CLIP models use a context length of 77
        self.context_length = getattr(
            opts, "dataset.multi_modal_img_text.context_length", 77
        )

        # for sharing padding index across the entire cvnets framework, we will
        # use a special variable "dataset.padding_index". The default value is set
        # to 0. If you need to override the default value, then use
        setattr(opts, "dataset.padding_index", None)
        self.padding_index = getattr(opts, "dataset.padding_index", None)

        # Because padding index does not exist in vocab, we add 0 for padding index.
        # So, we add 1 to total vocab size
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
            opts, "dataset.collate_fn_name_train", "multi_modal_img_text_collate_fn"
        )
        setattr(opts, "dataset.collate_fn_name_val", "multi_modal_img_text_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "multi_modal_img_text_collate_fn")

        self.zero_shot_dataset = self.get_zero_shot_dataset(*args, **kwargs)
        self.cached_zero_shot_captions = None

        # Path where we will download data
        self.cache_loc = os.path.join(self.root, ".img_text_tar_cache")
        os.makedirs(self.cache_loc, exist_ok=True)

        self.dataset = self.get_dataset(*args, **kwargs)

    def get_zero_shot_dataset(self, *args, **kwargs):
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

    def get_dataset(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        return parser

    def __len__(self):
        raise NotImplementedError

    #        if self.zeros_shot_dataset is not None:
    #            return len(self.zeros_shot_dataset)
    #        return len(self.dataset)

    def _transform_text(self, text_tensor: Tensor) -> Tuple[Tensor, int]:
        captions_tensor = torch.zeros(size=(self.context_length,), dtype=torch.long)

        text_len = text_tensor.shape[0]
        if text_len > self.context_length:
            text_tensor = text_tensor[: self.context_length]
            text_tensor[-1] = self.text_tokenizer.get_eot_token()
            text_len = self.context_length
        captions_tensor[:text_len] = text_tensor[:text_len]
        return captions_tensor, text_len

    def _training_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
            Training data augmentation methods.
                Image --> RandomResizedCrop --> RandomHorizontalFlip --> Optional(AutoAugment or RandAugment)
                --> Tensor --> Optional(RandomErasing) --> Optional(MixUp) --> Optional(CutMix)

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
            # T.RandomHorizontalFlip(opts=self.opts),
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

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Validation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _process_img_caption(
        self, input_img, captions_str, img_transform_fn, zero_shot: bool
    ) -> Tuple[Tensor, Tensor, int]:
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
                self.text_tokenizer(_caption_preprocessing(captions_str))
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
                            self.text_tokenizer(
                                _caption_preprocessing(captions_str_i_j)
                            )
                        )
                        captions_tensor_i.append(seq)
                        max_seq_len = max(max_seq_len, seq_len)
                    captions_tensor_i = torch.stack(captions_tensor_i, dim=0)
                    captions_tensor.append(captions_tensor_i)
                elif isinstance(captions_str_i, str):
                    # captions_str is [Num_templates_per_image]
                    seq, seq_len = self._transform_text(
                        self.text_tokenizer(_caption_preprocessing(captions_str_i))
                    )
                    captions_tensor.append(seq)
                    max_seq_len = max(max_seq_len, seq_len)
                else:
                    raise NotImplementedError
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

    def get_zero_shot_pair(self, img_index):
        img_path, captions_str, class_label = self.zero_shot_dataset(img_index)
        input_img = self.read_image_pil(img_path)
        return input_img, captions_str, class_label

    def get_dataset_pair(self, img_index):
        raise NotImplementedError

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

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
            img_tensor, captions_tensor, max_seq_len = self._process_img_caption(
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

    def extra_transform_repr(self):
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "img_transforms={}".format(transforms_str)

    def __repr__(self):
        return "{}(\n\troot={}\n\tis_training={}\n\tzero_shot={}\n\tn_samples={}\n\t{}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            self.zero_shot_dataset,
            self.__len__(),
            self.extra_transform_repr()
        )


def _caption_preprocessing(caption: str) -> str:
    # captions may contain HTML tokens. Remove them
    html_re = re.compile("<.*?>")
    caption = urllib.parse.unquote(str(caption))
    caption = caption.replace("+", " ")
    caption = re.sub(html_re, "", str(caption))
    # remove the next line
    caption = caption.strip("\n")
    # remove unwanted spaces
    caption = re.sub(" +", " ", caption)

    caption = ftfy.fix_text(caption)
    return caption.strip().lower()


@register_collate_fn(name="multi_modal_img_text_collate_fn")
def multi_modal_img_text_collate_fn(batch: List, opts) -> Dict:
    images = []
    text_tokens = []
    padding_mask = []
    labels = []

    truncate_seq_len = getattr(
        opts, "dataset.multi_modal_img_text.trunc_seq_len", False
    )

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
        if zero_shot:
            # For zero-shot, all text captions are the same
            # so, we only aggregate for one batch element
            if i == 0:
                text_tokens.append(text_data)
                if pad_mask is not None:
                    padding_mask.append(pad_mask)
        else:
            text_tokens.append(text_data)
            if pad_mask is not None:
                padding_mask.append(pad_mask)

    images = torch.stack(images, dim=0)
    text_tokens = torch.stack(text_tokens, dim=0)

    # truncate tokens based on the max. seq length
    if not truncate_seq_len:
        max_seq_len_in_batch = -1
    text_tokens = text_tokens[..., :max_seq_len_in_batch]

    if len(padding_mask) != 0:
        padding_mask = torch.stack(padding_mask, dim=0)
        padding_mask = padding_mask[..., :max_seq_len_in_batch]
    else:
        padding_mask = None

    labels = torch.tensor(labels, dtype=torch.long)

    channels_last = getattr(opts, "common.channels_last", False)
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

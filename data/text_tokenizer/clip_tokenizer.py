#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import Tensor
from torchtext.transforms import CLIPTokenizer

from data.text_tokenizer import TOKENIZER_REGISTRY, BaseTokenizer
from utils import logger
from utils.download_utils import get_local_path


@TOKENIZER_REGISTRY.register(name="clip")
class ClipTokenizer(BaseTokenizer):
    def __init__(self, opts, *args, **kwargs):
        merges_path = getattr(opts, "text_tokenizer.clip.merges_path", None)
        if merges_path is None:
            logger.error(
                "Please specify BPE merge file using --text-tokenizer.clip.merges-path argument"
            )

        # DDP case is handled internally
        merges_path = get_local_path(opts, path=merges_path)

        encoder_json_path = getattr(opts, "text_tokenizer.clip.encoder_json_path", None)
        if encoder_json_path is None:
            logger.error(
                "Please specify Encoder JSON file using --text-tokenizer.clip.encoder-json-path argument"
            )

        encoder_json_path = get_local_path(opts, path=encoder_json_path)

        super().__init__(opts, *args, **kwargs)
        self.tokenizer = CLIPTokenizer(
            merges_path=merges_path, encoder_json_path=encoder_json_path
        )
        # BPE encodings is a dict, where  keys are tokens and values are token_ids
        self.bpe_encodings = self.tokenizer.bpe.bpe_encoder_

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--text-tokenizer.clip.merges-path",
            type=str,
            default=None,
            help="Path to bpe merges file.",
        )

        group.add_argument(
            "--text-tokenizer.clip.encoder-json-path",
            type=str,
            default=None,
            help="Optional, path to BPE encoder json file. When specified, this is used to infer num_merges.",
        )
        return parser

    def get_vocab_size(self):
        return len(self.bpe_encodings)

    def get_encodings(self):
        return self.bpe_encodings

    def get_eot_token(self):
        return int(self.tokenizer("<|endoftext|>")[0])

    def get_sot_token(self):
        return int(self.tokenizer("<|startoftext|>")[0])

    def forward(self, input_sentence: str, *args, **kwargs) -> Tensor:
        # add start and eos tokens to input sentence
        input_sentence = "<|startoftext|> " + input_sentence + " <|endoftext|>"
        # tokenizer returns indices as a string
        tokenized_sentence = self.tokenizer(input_sentence)
        # convert string to int and then create a tensor
        tokenized_sentence = torch.tensor(
            [int(cap) for cap in tokenized_sentence], dtype=torch.long
        )
        return tokenized_sentence

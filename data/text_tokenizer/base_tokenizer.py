#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn
from typing import Any
import argparse


class BaseTokenizer(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )

        group.add_argument(
            "--text-tokenizer.name",
            type=str,
            default=None,
            help="Name of the text tokenizer.",
        )

        return parser

    def get_vocab_size(self):
        raise NotImplementedError

    def get_eot_token(self):
        raise NotImplementedError

    def get_sot_token(self):
        raise NotImplementedError

    def get_encodings(self):
        raise NotImplementedError

    def forward(self, input_sentence: Any, *args, **kwargs) -> Any:
        raise NotImplementedError
